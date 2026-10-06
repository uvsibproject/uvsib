from collections import defaultdict
from datetime import datetime, timezone
from ase.io import jsonio
from aiida.engine import WorkChain
from aiida.orm import Dict, List, Str
from aiida.plugins import WorkflowFactory
from uvsib.db.tables import DBSurfaceMLAdsorbate
from uvsib.db.utils import query_by_columns, add_akmc_events, merge_row_attributes
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings


def _akmc_settings():
    return settings.inputs.get("akmc", {})

def _surface_bound_adsorbate(name):
    return isinstance(name, str) and name.startswith("*") and name != "*"

class AKMCWorkChain(WorkChain):
    """Adaptive kinetic Monte Carlo workchain using MLIP dimer searches."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.select_minima,
            cls.run_akmc,
            cls.inspect_akmc,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The AKMC calculation did not finish successfully.")
        spec.exit_code(301, "ERROR_NO_MINIMA_FOUND", message="No adsorbate minima were found for AKMC.")

    def setup(self):
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.reaction_path = self.inputs.reaction_path.value
        cfg = _akmc_settings()
        self.ctx.max_paths = int(cfg.get("max_paths", 3))
        self.ctx.records = []
        self.report(
            f"Running AKMCWorkChain for {self.ctx.chemical_formula}. "
            f"Reaction: {self.ctx.reaction}, reaction_path: {self.ctx.reaction_path}"
        )

    def select_minima(self):
        rows = query_by_columns(
            DBSurfaceMLAdsorbate,
            {
                "composition": self.ctx.chemical_formula,
                "reaction": self.ctx.reaction,
                "reaction_path": self.ctx.reaction_path,
            },
        )
        rows = sorted(rows, key=lambda row: row.eta)

        # Each row is one complete adsorption path (one surface/site holding
        # *O, *OH, *OOH, ...). Keep the max_paths lowest-eta rows and run AKMC
        # on every surface-bound minimum in them, so each selected path is
        # covered end to end and every intermediate is sampled equally.
        records = []
        n_paths = 0
        for row in rows:
            adsorb_set = row.adsorb_set or {}
            row_records = []
            for atoms_json in adsorb_set.get("structures", []):
                atoms = jsonio.decode(atoms_json)
                adsorbate = atoms.info.get("adsorbate")
                if not _surface_bound_adsorbate(adsorbate):
                    continue
                row_records.append({
                    "row_id": row.id,
                    "surface_id": row.surface_id,
                    "eta": float(row.eta),
                    "adsorbate": adsorbate,
                    "site_type": row.site_type,
                    "ads_coord": row.ads_coord,
                    "reaction": row.reaction,
                    "reaction_path": row.reaction_path,
                    "atoms_json": atoms_json,
                })
            if not row_records:
                continue
            records.extend(row_records)
            n_paths += 1
            if n_paths >= self.ctx.max_paths:
                break

        if not records:
            self.report("No relaxed surface-bound adsorbate minima found for AKMC.")
            return self.exit_codes.ERROR_NO_MINIMA_FOUND

        self.ctx.records = records
        best = records[0]
        self.report(
            f"Selected {len(records)} AKMC minima from the {n_paths} lowest-eta "
            f"adsorption paths (rows {sorted({r['row_id'] for r in records})}); "
            f"best eta={best['eta']:.3f} eV on surface {best['surface_id']}."
        )

    def run_akmc(self):
        cfg = _akmc_settings()
        ml_model = cfg.get("model", settings.inputs["adsorbates"]["model"])
        model, model_path, device = get_model_device(ml_model)
        head = cfg.get("head", settings.inputs["adsorbates"].get("head"))

        Workflow = WorkflowFactory(ml_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(list=self.ctx.records)
        builder.local_label = Str(f"AKMC: {self.ctx.reaction_path} on {self.ctx.chemical_formula}")
        builder.code = get_code(ml_model)
        builder.job_info = Dict(dict={
            "job_type": "akmc",
            "ML_model": ml_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": head,
            "device": device,
            "fmax": float(cfg.get("fmax", settings.inputs["adsorbates"].get("fmax", 0.05))),
            "dimer_fmax": float(cfg.get("dimer_fmax", 0.08)),
            "relax_steps": int(cfg.get("relax_steps", settings.inputs["adsorbates"].get("max_steps", 200))),
            "dimer_steps": int(cfg.get("dimer_steps", 120)),
            "searches_per_minimum": int(cfg.get("searches_per_minimum", 4)),
            "temperature": float(cfg.get("temperature", 300.0)),
            "prefactor": float(cfg.get("prefactor", 1.0e13)),
            "dimer_displacement": float(cfg.get("dimer_displacement", 0.05)),
            "product_displacement": float(cfg.get("product_displacement", 0.15)),
            "maxstep": float(cfg.get("maxstep", 0.1)),
            "seed": int(cfg.get("seed", 13)),
        })
        future = self.submit(builder)
        self.to_context(akmc=future)

    def inspect_akmc(self):
        wch = self.ctx.akmc
        if not wch.is_finished_ok:
            self.report("AKMC MLIP sub-workchain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        output = wch.outputs.output_dict.get_dict()
        events = output.get("events", [])
        cfg = _akmc_settings()
        seed = int(cfg.get("seed", 13))

        events_by_row = {}
        for event in events:
            events_by_row.setdefault(event["source_row_id"], []).append(event)

        db_events = [
            {
                "source_row_id": event["source_row_id"],
                "composition": self.ctx.chemical_formula,
                "reaction": self.ctx.reaction,
                "reaction_path": self.ctx.reaction_path,
                "source_adsorbate": event.get("source_adsorbate"),
                "source_site_type": event.get("source_site_type"),
                "source_eta": event.get("source_eta"),
                "attempt": int(event.get("attempt", 0)),
                "initial_energy": event["initial_energy"],
                "saddle_energy": event["saddle_energy"],
                "final_energy": event["final_energy"],
                "barrier": event["barrier"],
                "reverse_barrier": event["reverse_barrier"],
                "rate": event["rate"],
                "temperature": event["temperature"],
                "prefactor": event["prefactor"],
                "seed": seed,
                "kmc_selected": bool(event.get("kmc_selected", False)),
                "kmc_time_increment": event.get("kmc_time_increment"),
                "final_state_key": event.get("final_state_key"),
                "initial_atoms_json": event["initial_atoms_json"],
                "saddle_atoms_json": event["saddle_atoms_json"],
                "final_atoms_json": event["final_atoms_json"],
            }
            for event in events
        ]
        add_akmc_events(db_events)

        run_time = datetime.now(timezone.utc).isoformat()
        for row_id, row_events in events_by_row.items():
            merge_row_attributes(DBSurfaceMLAdsorbate, row_id, {
                "akmc_last_run": {
                    "num_events": len(row_events),
                    "ctime": run_time,
                },
            })

        self.ctx.num_events = int(output.get("num_events", 0))
        self.ctx.num_rejected = int(output.get("num_rejected", 0))

    def final_report(self):
        self.report(
            f"AKMCWorkChain for {self.ctx.chemical_formula}: "
            f"{self.ctx.num_events} event(s), {self.ctx.num_rejected} rejected attempt(s)."
        )

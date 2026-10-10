"""OpticalScreenWorkChain -- no-DFT light-harvesting screen.

A mandatory ``MainWorkChain`` step (``step_status["optical_screen"]``), run after
the phase-diagram / SQS stages and before ``SurfaceBuilderWorkChain``, gated by
``settings.OPTICAL_SCREEN_ENABLED``; a failure stops the MainWorkChain. For the
composition's selected ML bulk structures it:

1. submits one ``ElectronicWorkChain`` job that predicts, without any DFT, the
   band gap (pretrained ML property models) and the Butler--Ginley / Mulliken
   band edges (see ``codes/files/electronic.py``);
2. adds the photocatalytic *straddle* verdict for every implemented
   (reaction, pathway) -- does the gap bracket that reaction's redox couple
   with margin? -- using ``redox_couples`` (single source of truth: the CHE
   calculators' own ``equilibrium_potential`` values);
3. writes the combined payload to ``DBStructureVersion.band_info`` for the
   version PhaseDiagramMLWorkChain ranked (method == ``ml_bulk_model``).

``pipeline_report.py`` then surfaces gap / edges / straddle per bulk.

Config (``input.yaml`` ``optical_screen:`` block, all optional)::

    optical_screen:
      enabled: true
      models: [megnet_mfi]        # gap models; see codes/files/electronic.py
      megnet_fidelity: 2          # 0 PBE, 1 GLLB-SC, 2 HSE, 3 SCAN
      gap_min: 1.4               # visible-light window (eV), label only
      gap_max: 3.1
      pH: 0.0
      straddle_margin: 0.2       # required head-room per band edge (V)
"""

from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory

from uvsib.db.tables import DBComposition
from uvsib.db.utils import query_by_columns, query_structure, update_structure_band_info
from uvsib.workchains.utils import get_code
from uvsib.workchains.redox_couples import all_couples, straddle_verdict
from uvsib.workflows import settings


def _config():
    cfg = settings.inputs.get("optical_screen", {}) or {}
    return {
        "models": list(cfg.get("models", ["megnet_mfi"])),
        "megnet_fidelity": int(cfg.get("megnet_fidelity", 2)),
        "gap_min": float(cfg.get("gap_min", 1.4)),
        "gap_max": float(cfg.get("gap_max", 3.1)),
        "pH": float(cfg.get("pH", 0.0)),
        "straddle_margin": float(cfg.get("straddle_margin", 0.2)),
    }


def _selected_bulks(chemical_formula):
    """``([{"uuid", "structure"}, ...], ml_bulk_model)`` for the composition's
    ML bulk selection (same source SurfaceBuilderWorkChain reads)."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return [], None
    stable = rows[0].stable_struct or {}
    model = stable.get("ml_bulk_model") or settings.inputs["bulk_relax"]["model"]
    payload = []
    for uuid_str in stable.get("ml_uuid_list", []) or []:
        versions = query_structure({"uuid": uuid_str}, method=model)
        if versions:
            payload.append({"uuid": uuid_str, "structure": versions[0].structure})
    return payload, model


def _straddle_block(edges_rhe, margin):
    """``{reaction: {pathway: straddle_verdict(...)}}`` for every implemented
    (reaction, pathway), from the band edges (V vs RHE)."""
    cb, vb = edges_rhe["cb"], edges_rhe["vb"]
    block = {}
    for reaction, by_path in all_couples().items():
        block[reaction] = {}
        for path, couple in by_path.items():
            verdict = straddle_verdict(cb, vb, couple["u_red"], couple["u_ox"], margin)
            verdict["role"] = couple["role"]
            verdict["label"] = couple["label"]
            block[reaction][path] = verdict
    return block


class OpticalScreenWorkChain(WorkChain):
    """No-DFT band gap + band-edge + straddle screen for the ML bulk selection."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_screen,
            cls.inspect_screen,
            cls.store_results,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The electronic screen did not finish successfully")
        spec.exit_code(301, "ERROR_NO_STRUCTURES_FOUND", message="No ML bulk selection to screen for the given formula")

    def setup(self):
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.cfg = _config()
        self.ctx.payload, self.ctx.model = _selected_bulks(self.ctx.chemical_formula)
        self.ctx.stored = 0
        self.report(f"Running OpticalScreenWorkChain for {self.ctx.chemical_formula} "
                    f"on {len(self.ctx.payload)} ML bulk structure(s); models={self.ctx.cfg['models']}")
        if not self.ctx.payload:
            self.report(f"No ML bulk selection found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND

    def run_screen(self):
        Workflow = WorkflowFactory("electronic")
        builder = Workflow.get_builder()
        builder.input_structures = List(list=self.ctx.payload)
        builder.code = get_code("Electronic")
        builder.local_label = Str(f"optical screen: {self.ctx.chemical_formula}")
        cfg = self.ctx.cfg
        builder.job_info = Dict(dict={
            "models": cfg["models"],
            "megnet_fidelity": cfg["megnet_fidelity"],
            "gap_min": cfg["gap_min"],
            "gap_max": cfg["gap_max"],
            "pH": cfg["pH"],
        })
        self.to_context(screen=self.submit(builder))

    def inspect_screen(self):
        wch = self.ctx.screen
        if not wch.is_finished_ok:
            self.report("Electronic screen sub-workchain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        output = wch.outputs.output_dict.get_dict()
        self.ctx.results = output.get("results", [])
        self.ctx.screen_status = output.get("status")
        if self.ctx.screen_status != "ok":
            self.report("Electronic screen ran but no ML gap model was importable "
                        "on the Electronic code environment (status='unavailable'); "
                        "band edges / straddle not computed.")

    def store_results(self):
        margin = self.ctx.cfg["straddle_margin"]
        for result in self.ctx.results:
            band_info = result["band_info"]
            edges = band_info.get("band_edges_vs_rhe_V")
            if edges:
                band_info["straddle"] = _straddle_block(edges, margin)
                band_info["straddle_margin_V"] = margin
            if update_structure_band_info(result["uuid"], self.ctx.model, band_info):
                self.ctx.stored += 1
            else:
                self.report(f"No {self.ctx.model} version row for {result['uuid']}; band_info not stored.")

    def final_report(self):
        self.report(f"OpticalScreenWorkChain for {self.ctx.chemical_formula}: "
                    f"band_info written for {self.ctx.stored}/{len(self.ctx.results)} structure(s).")

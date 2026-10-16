"""PhononScreenWorkChain -- MLIP phonons for the *predicted* ML bulk selection.

A ``MainWorkChain`` step (``step_status["phonon"]``, gated by
``settings.PHONON_ENABLED``, on by default) run after the phase-diagram stages. For every bulk
in the composition's ML selection (``DBComposition.stable_struct["ml_selection"]``)
whose ``DBStructureVersion.source`` is a structure-prediction source
(``csp`` / ``generated``) -- Materials Project structures (``MPDB_*``) are
skipped -- it:

1. submits ``PhononWorkChain`` jobs (``chunk_size`` structures each, in
   parallel) that tight-relax, run phonopy finite displacements with the MLIP,
   and compute the harmonic (or, with ``harmonic_only: false``, QHA) free
   energy (see ``codes/files/phonon.py``);
2. flags dynamically unstable structures (imaginary modes off Gamma);
3. writes one ``DBPhonon`` row per bulk.

Nothing downstream drops a structure on this basis, but the stage is
mandatory when enabled: if any phonon calculation fails (a whole
``PhononWorkChain`` or a single structure inside it) the workchain exits 300
and the MainWorkChain stops. Results that did succeed are still stored, and
with ``skip_existing`` a rerun only redoes the failed structures.
``pipeline_report.py`` renders it as the "Lattice dynamics" block.

Config (``input.yaml`` ``phonon:`` block, all optional except ``enabled``)::

    phonon:
      enabled: true                # default true
      model: SevenNet              # default: bulk_relax.model; must be conservative (F = -dE/dx)
      head: omat24                 # default: bulk_relax.head
      include_above_threshold: true  # also phonon the min_n_return fallback bulk
      skip_existing: true          # do not redo bulks with an ok DBPhonon row for this model
      harmonic_only: true          # false -> QHA over volume_scales (~n_volumes x cost)
      volume_scales: [0.97, 0.99, 1.0, 1.01, 1.03]
      fmax: 0.001                  # eV/A, tight relax before the displacements
      max_steps: 500
      min_supercell_length: 8.0    # A (diagonal supercell, at most 3x per axis)
      displacement: 0.01           # A
      mesh: 20                     # q-mesh per axis for DOS / thermal properties
      t_min: 0.0
      t_max: 1500.0
      t_step: 50.0
      imaginary_tol_THz: 0.1       # modes below -tol (off Gamma) count as imaginary
      chunk_size: 10               # structures per PhononWorkChain job
"""

from aiida.engine import WorkChain, if_
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory

from uvsib.db.tables import DBComposition, DBPhonon
from uvsib.db.utils import query_by_columns, query_structure, upsert_phonon
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

# DBStructureVersion.source values written by structure prediction
# (CSPWorkChain -> "csp", GeneratorWorkChain -> "generated").
PREDICTED_SOURCES = ("csp", "generated")
REFERENCE_TEMPERATURE = 300.0


def _config():
    cfg = settings.inputs.get("phonon", {}) or {}
    bulk = settings.inputs["bulk_relax"]
    model = str(cfg.get("model") or bulk["model"])
    head = cfg.get("head", bulk.get("head") if model == bulk["model"] else None)
    scales = cfg.get("volume_scales", [0.97, 0.99, 1.0, 1.01, 1.03])
    if isinstance(scales, str):
        scales = [float(x) for x in scales.split(",")]
    if bool(cfg.get("harmonic_only", True)):
        scales = [1.0]
    return {
        "model": model,
        "head": head,
        "include_above_threshold": bool(cfg.get("include_above_threshold", True)),
        "skip_existing": bool(cfg.get("skip_existing", True)),
        "volume_scales": [float(x) for x in scales],
        "fmax": float(cfg.get("fmax", 1e-3)),
        "max_steps": int(cfg.get("max_steps", 500)),
        "min_supercell_length": float(cfg.get("min_supercell_length", 8.0)),
        "displacement": float(cfg.get("displacement", 0.01)),
        "mesh": int(cfg.get("mesh", 20)),
        "t_min": float(cfg.get("t_min", 0.0)),
        "t_max": float(cfg.get("t_max", 1500.0)),
        "t_step": float(cfg.get("t_step", 50.0)),
        "imaginary_tol_THz": float(cfg.get("imaginary_tol_THz", 0.1)),
        "chunk_size": max(1, int(cfg.get("chunk_size", 10))),
    }


def _done_uuids(chemical_formula, model):
    """uuids that already have an ok DBPhonon row for ``model``."""
    return {str(row.structure_uuid)
            for row in query_by_columns(DBPhonon, {"composition": chemical_formula,
                                                   "phonon_model": model})
            if row.status == "ok"}


def _predicted_bulks(chemical_formula, cfg):
    """``([{"uuid", "structure"}, ...], {uuid: info}, n_mp_skipped, n_done_skipped)``
    for the *predicted* bulks PhaseDiagramMLWorkChain kept after the
    E_above_hull screen. ``info`` carries the ml_selection entry and the source."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return [], {}, 0, 0
    stable = rows[0].stable_struct or {}
    bulk_model = stable.get("ml_bulk_model") or settings.inputs["bulk_relax"]["model"]
    done = _done_uuids(chemical_formula, cfg["model"]) if cfg["skip_existing"] else set()

    payload, info = [], {}
    n_mp, n_done = 0, 0
    for entry in stable.get("ml_selection", []) or []:
        if entry.get("selected_above_threshold") and not cfg["include_above_threshold"]:
            continue
        versions = query_structure({"uuid": entry["uuid"]}, method=bulk_model)
        if not versions:
            continue
        version = versions[0]
        if version.source not in PREDICTED_SOURCES:
            n_mp += 1
            continue
        if entry["uuid"] in done:
            n_done += 1
            continue
        info[entry["uuid"]] = {"entry": entry, "source": version.source,
                               "ml_bulk_model": bulk_model}
        payload.append({"uuid": entry["uuid"], "structure": version.structure})
    return payload, info, n_mp, n_done


def _at_temperature(temperatures, values, target):
    """``(value, T)`` at the grid temperature closest to ``target``; ``(None,
    None)`` if the grid does not reach it."""
    if not temperatures or not values or max(temperatures) < target - 1e-6:
        return None, None
    idx = min(range(len(temperatures)), key=lambda i: abs(temperatures[i] - target))
    return values[idx], temperatures[idx]


class PhononScreenWorkChain(WorkChain):
    """MLIP phonon / dynamical-stability screen for the predicted ML bulk selection."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            if_(cls.should_run_phonons)(
                cls.run_phonons,
                cls.inspect_phonons,
                cls.store_results,
            ),
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="One or more phonon calculations failed")

    def setup(self):
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.cfg = _config()
        self.ctx.payload, self.ctx.info, n_mp, n_done = _predicted_bulks(
            self.ctx.chemical_formula, self.ctx.cfg)
        self.ctx.results = []
        self.ctx.stored = 0
        self.ctx.n_failed = 0
        self.report(f"Running PhononScreenWorkChain ({self.ctx.cfg['model']}) for "
                    f"{self.ctx.chemical_formula}: {len(self.ctx.payload)} predicted bulk(s) "
                    f"to phonon, {n_mp} Materials Project bulk(s) skipped, "
                    f"{n_done} already done.")

    def should_run_phonons(self):
        return bool(self.ctx.payload)

    def _code_and_device(self):
        """The dedicated ``Phonon`` code if config.yaml has one, else the model's own code."""
        codes = settings.configs["codes"]
        model_name, model_path, device = get_model_device(self.ctx.cfg["model"])
        phonon_block = codes.get("Phonon") or {}
        if phonon_block.get("code_string"):
            device = phonon_block["job_script"].get("device", device)
            return get_code("Phonon"), model_name, model_path, device
        return get_code(self.ctx.cfg["model"]), model_name, model_path, device

    def run_phonons(self):
        cfg = self.ctx.cfg
        code, model_name, model_path, device = self._code_and_device()
        job_info = {
            "ML_model": cfg["model"],
            "model_name": model_name,
            "model_path": model_path,
            "model_head": cfg["head"],
            "device": device,
            **{k: cfg[k] for k in ("fmax", "max_steps", "min_supercell_length",
                                   "displacement", "mesh", "t_min", "t_max",
                                   "t_step", "volume_scales", "imaginary_tol_THz")},
        }
        Workflow = WorkflowFactory("phonon")
        size = cfg["chunk_size"]
        chunks = [self.ctx.payload[i:i + size] for i in range(0, len(self.ctx.payload), size)]
        self.ctx.chunk_uuids = {}
        for i, chunk in enumerate(chunks):
            builder = Workflow.get_builder()
            builder.input_structure = List(list=chunk)
            builder.code = code
            builder.job_info = Dict(dict=job_info)
            builder.local_label = Str(f"phonon screen: {self.ctx.chemical_formula} "
                                      f"({i + 1}/{len(chunks)})")
            key = f"phonon_{i}"
            self.ctx.chunk_uuids[key] = [item["uuid"] for item in chunk]
            self.to_context(**{key: self.submit(builder)})

    def inspect_phonons(self):
        for key, uuids in self.ctx.chunk_uuids.items():
            wch = self.ctx[key]
            if not wch.is_finished_ok:
                self.report(f"PhononWorkChain {key} failed (exit {wch.exit_status}); "
                            f"{len(uuids)} structure(s) marked failed.")
                self.ctx.results += [{"uuid": u, "error": f"PhononWorkChain exit {wch.exit_status}"}
                                     for u in uuids]
                continue
            output = wch.outputs.output_dict.get_dict()
            self.ctx.results += output.get("results", [])
            self.ctx.run_params = output.get("params", {})
        # a structure the runner never returned counts as failed too
        returned = {r.get("uuid") for r in self.ctx.results}
        self.ctx.results += [{"uuid": item["uuid"], "error": "no result returned by the phonon runner"}
                             for item in self.ctx.payload if item["uuid"] not in returned]

    def store_results(self):
        cfg = self.ctx.cfg
        for result in self.ctx.results:
            uuid = result.get("uuid")
            info = self.ctx.info.get(uuid)
            if info is None:
                continue
            values = {
                "structure_uuid": uuid,
                "composition": self.ctx.chemical_formula,
                "phonon_model": cfg["model"],
                "source": info["source"],
                "ehull": info["entry"].get("ehull"),
            }
            if "error" in result:
                self.ctx.n_failed += 1
                values.update({
                    "status": "failed",
                    "dynamically_stable": None,
                    "min_frequency_THz": None,
                    "n_imaginary_qpoints": None,
                    "e0": None,
                    "free_energy_300K": None,
                    "attributes": {"error": result["error"], "head": cfg["head"],
                                   "ml_bulk_model": info["ml_bulk_model"]},
                })
            else:
                temps = result.get("temperatures") or []
                f_ref, t_ref = _at_temperature(
                    temps, result.get("free_energy_correction_eV_per_atom"), REFERENCE_TEMPERATURE)
                s_ref, _ = _at_temperature(
                    temps, result.get("entropy_kB_per_atom"), REFERENCE_TEMPERATURE)
                values.update({
                    "status": "ok",
                    "dynamically_stable": not bool(result.get("imaginary_modes")),
                    "min_frequency_THz": result.get("min_frequency_THz"),
                    "n_imaginary_qpoints": result.get("n_imaginary_qpoints"),
                    "e0": result.get("E0_eV_per_atom"),
                    "free_energy_300K": f_ref,
                    "attributes": {
                        "formula": result.get("formula"),
                        "head": cfg["head"],
                        "ml_bulk_model": info["ml_bulk_model"],
                        "selected_above_threshold": bool(
                            info["entry"].get("selected_above_threshold", False)),
                        "reference_temperature": t_ref,
                        "entropy_300K_kB_per_atom": s_ref,
                        "temperatures": temps,
                        "free_energy_correction_eV_per_atom": result.get("free_energy_correction_eV_per_atom"),
                        "G_qha_eV_per_atom": result.get("G_qha_eV_per_atom"),
                        "entropy_kB_per_atom": result.get("entropy_kB_per_atom"),
                        "volume_per_atom_A3": result.get("volume_per_atom_A3"),
                        "imaginary_modes_any_volume": result.get("imaginary_modes_any_volume"),
                        "max_frequency_THz": result.get("max_frequency_THz"),
                        "n_qpoints": result.get("n_qpoints"),
                        "supercell_matrix": result.get("supercell_matrix"),
                        "n_atoms": result.get("n_atoms"),
                        "n_atoms_supercell": result.get("n_atoms_supercell"),
                        "n_displacements": result.get("n_displacements"),
                        "n_volumes": result.get("n_volumes"),
                        "relax_converged": result.get("relax_converged"),
                        "run_params": self.ctx.get("run_params", {}),
                    },
                })
            upsert_phonon(values)
            self.ctx.stored += 1

        if self.ctx.n_failed:
            failed = [r["uuid"] for r in self.ctx.results if "error" in r]
            self.report(f"Phonon calculations failed for {self.ctx.n_failed}/"
                        f"{len(self.ctx.payload)} structure(s): {failed}")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def final_report(self):
        if not self.ctx.payload:
            self.report(f"PhononScreenWorkChain for {self.ctx.chemical_formula}: "
                        "no predicted bulk to phonon.")
            return
        self.report(f"PhononScreenWorkChain for {self.ctx.chemical_formula}: "
                    f"{self.ctx.stored}/{len(self.ctx.payload)} structure(s) stored, "
                    f"{self.ctx.n_failed} failed.")

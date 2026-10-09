"""SynthesizabilityScreenWorkChain -- CSLLM synthesizability / method / precursors.

A ``PhaseDiagramMLWorkChain`` branch (run after the optical screen, gated by
``settings.SYNTHESIZABILITY_ENABLED``). For every bulk in the composition's ML
selection -- the structures that passed the E_above_hull screen in
``store_stable_structs`` -- it:

1. submits one ``CSLLMWorkChain`` job that loads CSLLM once and
   predicts P(synthesizable), the synthesis method, and ranked precursor sets
   (see ``codes/files/csllm.py``);
2. turns the raw probabilities into a label with the configured thresholds;
3. writes one ``DBSynthesizability`` row per bulk.

Nothing downstream reads it to drop a structure, but the stage is mandatory
when enabled: a failure fails the phase diagram (exit 305). ``pipeline_report.py`` renders it as
the "Synthesizability" block.

Config (``input.yaml`` ``synthesizability:`` block, all optional except
``enabled``)::

    synthesizability:
      enabled: true
      include_above_threshold: true  # also score the min_n_return fallback bulk
      score_threshold: 0.5           # P(synthesizable) cut for the label only
      uncertainty_threshold: 0.8     # entropy (bits) above which the label is "uncertain"
      predict_method: true
      predict_precursors: true
      precursor_num_sets: 3          # beam-search precursor lists per formula
      precursor_max_elements: 3      # CSLLM precursors validated on binaries/ternaries
      max_new_tokens: 64
      symprec: 0.01                  # pyxtal tolerance for the material string
      dtype: auto                    # auto | bfloat16 | float16 | float32
"""

from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory
from pymatgen.core import Composition

from uvsib.db.tables import DBComposition
from uvsib.db.utils import query_by_columns, query_structure, upsert_synthesizability
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

MODEL_NAME = "CSLLM"


def _config():
    cfg = settings.inputs.get("synthesizability", {}) or {}
    return {
        "include_above_threshold": bool(cfg.get("include_above_threshold", True)),
        "score_threshold": float(cfg.get("score_threshold", 0.5)),
        "uncertainty_threshold": float(cfg.get("uncertainty_threshold", 0.8)),
        "predict_method": bool(cfg.get("predict_method", True)),
        "predict_precursors": bool(cfg.get("predict_precursors", True)),
        "precursor_num_sets": int(cfg.get("precursor_num_sets", 3)),
        "precursor_max_elements": int(cfg.get("precursor_max_elements", 3)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 64)),
        "symprec": float(cfg.get("symprec", 0.01)),
        "dtype": str(cfg.get("dtype", "auto")),
    }


def _selected_bulks(chemical_formula, include_above_threshold):
    """``([{"uuid", "structure"}, ...], {uuid: ml_selection entry}, ml_bulk_model)``
    for the bulks PhaseDiagramMLWorkChain kept after the E_above_hull screen."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return [], {}, None
    stable = rows[0].stable_struct or {}
    model = stable.get("ml_bulk_model") or settings.inputs["bulk_relax"]["model"]
    selection = {}
    payload = []
    for entry in stable.get("ml_selection", []) or []:
        if entry.get("selected_above_threshold") and not include_above_threshold:
            continue
        versions = query_structure({"uuid": entry["uuid"]}, method=model)
        if versions:
            selection[entry["uuid"]] = entry
            payload.append({"uuid": entry["uuid"], "structure": versions[0].structure})
    return payload, selection, model


def label_for(p_true, entropy_bits, cfg):
    """Report label from P(synthesizable) and its entropy -- display only."""
    if p_true is None:
        return None
    if entropy_bits is not None and entropy_bits > cfg["uncertainty_threshold"]:
        return "uncertain"
    return "synthesizable" if p_true >= cfg["score_threshold"] else "not synthesizable"


class SynthesizabilityScreenWorkChain(WorkChain):
    """CSLLM synthesizability / method / precursor screen for the ML bulk selection."""

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

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The synthesizability screen did not finish successfully")
        spec.exit_code(301, "ERROR_NO_STRUCTURES_FOUND", message="No ML bulk selection to screen for the given formula")

    def setup(self):
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.cfg = _config()
        self.ctx.payload, self.ctx.selection, self.ctx.model = _selected_bulks(
            self.ctx.chemical_formula, self.ctx.cfg["include_above_threshold"])
        self.ctx.stored = 0
        self.report(f"Running SynthesizabilityScreenWorkChain ({MODEL_NAME}) for "
                    f"{self.ctx.chemical_formula} on {len(self.ctx.payload)} ML bulk structure(s)")
        if not self.ctx.payload:
            self.report(f"No ML bulk selection found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND

    def run_screen(self):
        _, weights_dir, device = get_model_device(MODEL_NAME)
        cfg = self.ctx.cfg
        Workflow = WorkflowFactory("csllm")
        builder = Workflow.get_builder()
        builder.input_structures = List(list=self.ctx.payload)
        builder.code = get_code(MODEL_NAME)
        builder.local_label = Str(f"synthesizability screen: {self.ctx.chemical_formula}")
        builder.job_info = Dict(dict={
            "weights_dir": weights_dir,
            "device": device,
            "dtype": cfg["dtype"],
            "predict_method": cfg["predict_method"],
            "predict_precursors": cfg["predict_precursors"],
            "precursor_num_sets": cfg["precursor_num_sets"],
            "max_new_tokens": cfg["max_new_tokens"],
            "symprec": cfg["symprec"],
        })
        self.to_context(screen=self.submit(builder))

    def inspect_screen(self):
        wch = self.ctx.screen
        if not wch.is_finished_ok:
            self.report("Synthesizability sub-workchain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        output = wch.outputs.output_dict.get_dict()
        self.ctx.results = output.get("results", [])
        self.ctx.run_config = output.get("config", {})
        if output.get("status") != "ok":
            self.report("CSLLM could not be loaded on the CSLLM code environment "
                        "(status='unavailable'); no synthesizability stored.")

    def store_results(self):
        cfg = self.ctx.cfg
        for result in self.ctx.results:
            synth = result.get("synthesizability") or {}
            p_true = synth.get("p_true")
            entropy = synth.get("entropy_bits")
            method = result.get("method") or {}
            n_elements = len(Composition(result["formula"]).elements)
            entry = self.ctx.selection.get(result["uuid"], {})
            upsert_synthesizability({
                "structure_uuid": result["uuid"],
                "composition": self.ctx.chemical_formula,
                "synthesizability_model": MODEL_NAME,
                "synthesizability_score": p_true,
                "synthesizability_label": label_for(p_true, entropy, cfg),
                "synthesizability_uncertainty": entropy,
                "predicted_synthesis_method": method.get("label"),
                "predicted_precursors": result.get("precursors"),
                "in_domain": n_elements <= cfg["precursor_max_elements"],
                "ehull": entry.get("ehull"),
                "attributes": {
                    "formula": result["formula"],
                    "material_string": result.get("material_string"),
                    "spacegroup": result.get("spacegroup"),
                    "symmetry_tol": result.get("symmetry_tol"),
                    "label_mass": synth.get("label_mass"),
                    "method_probabilities": method.get("probabilities"),
                    "method_label_mass": method.get("label_mass"),
                    "selected_above_threshold": bool(entry.get("selected_above_threshold", False)),
                    "ml_bulk_model": self.ctx.model,
                    "thresholds": {"score": cfg["score_threshold"],
                                   "uncertainty_bits": cfg["uncertainty_threshold"],
                                   "precursor_max_elements": cfg["precursor_max_elements"]},
                    "run_config": self.ctx.run_config,
                    "notes": result.get("notes", []),
                },
            })
            self.ctx.stored += 1

    def final_report(self):
        self.report(f"SynthesizabilityScreenWorkChain for {self.ctx.chemical_formula}: "
                    f"{self.ctx.stored}/{len(self.ctx.payload)} structure(s) stored.")

"""
CSLLM (https://github.com/szl666/CSLLM, Song et al., Nat. Commun. 16, 6530
(2025)) synthesizability / synthesis-method / precursor prediction for a batch
of bulk structures -- no DFT, three LoRA adapters on one shared LLaMA3-8B base.

Weights (https://huggingface.co/zhilong777/csllm) are expected under
``--weights_dir`` as published there:

    <weights_dir>/llama3-8bf-hf/            base model (loaded ONCE)
    <weights_dir>/synthesis_llm_llama3/     LoRA adapter: synthesizable?   (structure)
    <weights_dir>/method_llm_llama3/        LoRA adapter: synthesis method (formula)
    <weights_dir>/precursor_llm_llama3/     LoRA adapter: precursors       (formula)

Prompts are the ones the adapters were fine-tuned on (CSLLM ``material_str.py``
/ ``cons_data.py``, LMFlow ``text_only`` format)::

    Input: Can this material structure be synthesized "<material string>"? \\n Output: True|False
    Input: How can this material structure be synthesized "<formula>"? \\n Output: solid_state|solution|solid_state&solution
    Input: How can this material structure be synthesized "<formula>"? \\n Output: ['SrO', 'Fe2O3', 'MoO3']

(``gui.py`` in the CSLLM repo wraps these in a different chat template; that is
NOT what the adapters saw in training, so it is not used here.)

Synthesizability and method are closed-label tasks, so instead of parsing free
generated text, every label's sequence probability is scored directly
(``label_probabilities``) -- this yields a calibrated-looking P(True) to store
as the score, an entropy-based uncertainty, and the method class
probabilities. Precursors are open-ended: beam search returns the top
``--precursor_num_sets`` lists, each parsed and element-checked against the
target (``check_precursors``).

Input (``input_structures.json``, staged via the ``file`` namespace):

    [{"uuid": <bulk structure uuid>, "structure": <pymatgen Structure.as_dict()>}, ...]

Output (``output.json``, parsed into ``output_dict`` by ``synthesizability_parser``):

    {"results": [{"uuid", "formula", "material_string", "spacegroup",
                  "synthesizability": {"p_true", "label_mass", "entropy_bits"},
                  "method": {"label", "probabilities", "label_mass"} | None,
                  "precursors": [{"precursors", "beam_score", ...checks}, ...] | None,
                  "notes": [...]}, ...],
     "config": {...},
     "status": "ok" | "unavailable"}      # "unavailable": weights/libraries not loadable
"""

import argparse
import ast
import json
import math
import os
import re

from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element

SYN_PROMPT = 'Input: Can this material structure be synthesized "{}"? \n Output:'
FORMULA_PROMPT = 'Input: How can this material structure be synthesized "{}"? \n Output:'
SYN_LABELS = ["True", "False"]
METHOD_LABELS = ["solid_state", "solution", "solid_state&solution"]

# Elements a precursor set may lack (supplied by the atmosphere / a flux gas)
# or carry in excess (leave as volatile CO2 / H2O / NOx / NH3) without the set
# being flagged as element-inconsistent with the target.
ATMOSPHERE_ELEMENTS = {"O", "N", "H"}
VOLATILE_ELEMENTS = {"O", "C", "H", "N"}

_ADAPTERS = {
    "synthesis": "synthesis_llm_llama3",
    "method": "method_llm_llama3",
    "precursor": "precursor_llm_llama3",
}
_BASE = "llama3-8bf-hf"


# --------------------------------------------------------------------------- #
# CSLLM text representations
# --------------------------------------------------------------------------- #
def material_string(structure, tol):
    """CSLLM "material string" (``material_str.structure_to_str``):
    ``"<spg number> |a,b,c,alpha,beta,gamma| (El-<mult><letter>[x y z])->..."``.
    Lattice numbers are parsed from ``str(pyxtal.lattice)`` exactly as the
    training data was built, so the rounding matches."""
    from pyxtal import pyxtal

    px = pyxtal()
    px._from_pymatgen(structure, tol=tol)
    a, b, c, alpha, beta, gamma = [float(i.replace(" ", "")) for i in str(px.lattice).split(",")[:-1]]
    lattice_str = f"|{a:.3f},{b:.3f},{c:.3f},{alpha:.2f},{beta:.2f},{gamma:.2f}|"
    wyckoffs = [f"({site.specie}-{site.wp.multiplicity}{site.wp.letter}{site.position})"
                for site in px.atom_sites]
    return f"{px.group.number} {lattice_str} {'->'.join(wyckoffs)}", int(px.group.number), len(wyckoffs)


def material_string_with_fallback(structure, tols):
    """MLIP-relaxed cells carry small symmetry noise, and a too-tight
    tolerance does not fail -- it silently returns P1, a structure CSLLM never
    saw in that form. So every tolerance is tried and the most symmetric
    description (fewest Wyckoff sites) wins; ties go to the tightest
    tolerance. Returns ``(material string, space group number, tol used)``."""
    best = None
    last_exc = None
    for tol in sorted(tols):
        try:
            text, spg, n_sites = material_string(structure, tol)
        except Exception as exc:  # noqa: BLE001 - pyxtal raises assorted errors on noisy cells
            last_exc = exc
            continue
        if best is None or n_sites < best[2]:
            best = (text, spg, n_sites, tol)
    if best is None:
        raise ValueError(f"pyxtal could not symmetrize the structure: {last_exc}")
    return best[0], best[1], best[3]


# --------------------------------------------------------------------------- #
# scoring helpers (pure python -- no torch needed)
# --------------------------------------------------------------------------- #
def label_probabilities(seq_probs):
    """``{label: P(label tokens | prompt)}`` -> normalized class probabilities.

    A label that is a prefix of another (``solid_state`` vs
    ``solid_state&solution``) has its sequence probability include every
    longer continuation, so those are subtracted first. Returns
    ``(probabilities, label_mass)``; ``label_mass`` (sum of the exclusive
    probabilities before normalization) close to 1 means the model answers in
    the format it was trained on -- a low value is a red flag."""
    exclusive = {}
    for label, p in seq_probs.items():
        longer = sum(q for other, q in seq_probs.items() if other != label and other.startswith(label))
        exclusive[label] = max(p - longer, 0.0)
    mass = sum(exclusive.values())
    if mass <= 0.0:
        return {label: 1.0 / len(exclusive) for label in exclusive}, 0.0
    return {label: p / mass for label, p in exclusive.items()}, mass


def binary_entropy_bits(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def parse_precursor_list(text):
    """First ``[...]`` list in the generated text -> list of formula strings
    (``None`` if nothing parseable)."""
    match = re.search(r"\[[^\[\]]*\]", text)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, (list, tuple)):
        return None
    out = [str(v).strip() for v in value if str(v).strip()]
    return out or None


def check_precursors(precursors, target_formula):
    """Element-level consistency of one precursor set with the target -- not a
    stoichiometric balance. Every target element must come from some
    precursor (O/N/H may come from the atmosphere); precursor elements absent
    from the target must be volatile (O/C/H/N, e.g. carbonates, nitrates,
    hydroxides)."""
    target = {el.symbol for el in Composition(target_formula).elements}
    supplied = set()
    unparsable = []
    for formula in precursors:
        try:
            symbols = {el.symbol for el in Composition(formula).elements}
        except Exception:  # noqa: BLE001 - pymatgen raises several types on bad formulas
            unparsable.append(formula)
            continue
        # pymatgen accepts made-up symbols ("Xx") as DummySpecies
        if not all(Element.is_valid_symbol(sym) for sym in symbols):
            unparsable.append(formula)
            continue
        supplied |= symbols
    missing = sorted(target - supplied - ATMOSPHERE_ELEMENTS)
    extra = sorted(supplied - target - VOLATILE_ELEMENTS)
    return {
        "unparsable": unparsable,
        "missing_elements": missing,
        "extra_elements": extra,
        "element_consistent": not unparsable and not missing and not extra,
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def check_weights(weights_dir):
    """Fail with the actual missing files -- otherwise transformers treats a
    missing local directory as a Hub repo id and raises a misleading
    "Repo id must be in the form ..." error."""
    problems = []
    base_dir = os.path.join(weights_dir, _BASE)
    if not os.path.isdir(base_dir):
        problems.append(f"missing directory {base_dir}")
    else:
        files = os.listdir(base_dir)
        for name in ("config.json", "tokenizer.json"):
            if name not in files:
                problems.append(f"{base_dir}/{name} missing")
        if not any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            problems.append(f"no model weight shards (*.safetensors / *.bin) in {base_dir}")
    for adapter in _ADAPTERS.values():
        adapter_dir = os.path.join(weights_dir, adapter)
        if not os.path.isdir(adapter_dir):
            problems.append(f"missing directory {adapter_dir}")
            continue
        files = os.listdir(adapter_dir)
        if "adapter_config.json" not in files:
            problems.append(f"{adapter_dir}/adapter_config.json missing")
        if not any(f.startswith("adapter_model.") for f in files):
            problems.append(f"no adapter_model.bin / .safetensors in {adapter_dir}")
    if problems:
        raise FileNotFoundError("CSLLM weights incomplete: " + "; ".join(problems))


class CSLLM:
    """LLaMA3-8B base + the three CSLLM LoRA adapters, switched per task."""

    def __init__(self, weights_dir, device, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.torch = torch
        base_dir = os.path.join(weights_dir, _BASE)
        check_weights(weights_dir)
        if dtype == "auto":
            if device.startswith("cuda"):
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype = "float32"
        self.dtype = dtype
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_dir)
        base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=getattr(torch, dtype))
        base.to(device)
        self.model = PeftModel.from_pretrained(
            base, os.path.join(weights_dir, _ADAPTERS["synthesis"]), adapter_name="synthesis")
        for name in ("method", "precursor"):
            self.model.load_adapter(os.path.join(weights_dir, _ADAPTERS[name]), adapter_name=name)
        self.model.eval()

    def _use(self, adapter):
        self.model.set_adapter(adapter)

    def sequence_probs(self, adapter, prompt, labels):
        """``{label: P(" " + label | prompt)}`` -- product of the label tokens'
        conditional probabilities, from one forward pass per label."""
        torch = self.torch
        self._use(adapter)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        out = {}
        with torch.no_grad():
            for label in labels:
                label_ids = self.tokenizer(" " + label, add_special_tokens=False,
                                           return_tensors="pt").input_ids[0]
                ids = torch.cat([prompt_ids, label_ids]).unsqueeze(0).to(self.device)
                logits = self.model(input_ids=ids).logits[0].float()
                logprobs = torch.log_softmax(logits, dim=-1)
                start = prompt_ids.shape[0]
                total = 0.0
                for i, tok in enumerate(label_ids.tolist()):
                    total += float(logprobs[start + i - 1, tok])
                out[label] = math.exp(total)
        return out

    def generate(self, adapter, prompt, num_sets, max_new_tokens):
        """Beam-search continuations: ``[(text, length-normalized beam score), ...]``."""
        torch = self.torch
        self._use(adapter)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=max(num_sets, 1),
                num_return_sequences=max(num_sets, 1),
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        start = enc.input_ids.shape[1]
        scores = gen.sequences_scores.tolist() if getattr(gen, "sequences_scores", None) is not None \
            else [None] * len(gen.sequences)
        return [(self.tokenizer.decode(seq[start:], skip_special_tokens=True), score)
                for seq, score in zip(gen.sequences, scores)]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def predict_formula(llm, formula, cfg):
    """Method + precursors depend only on the formula, so they are computed
    once per distinct formula and shared by every polymorph."""
    prompt = FORMULA_PROMPT.format(formula)
    method = precursors = None
    if cfg["predict_method"]:
        probs, mass = label_probabilities(llm.sequence_probs("method", prompt, METHOD_LABELS))
        method = {
            "label": max(probs, key=probs.get),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "label_mass": round(mass, 4),
        }
    if cfg["predict_precursors"]:
        precursors = []
        for text, beam_score in llm.generate("precursor", prompt, cfg["precursor_num_sets"],
                                          cfg["max_new_tokens"]):
            parsed = parse_precursor_list(text)
            entry = {"precursors": parsed, "beam_score": beam_score, "raw": text.strip()[:300]}
            if parsed:
                entry.update(check_precursors(parsed, formula))
            precursors.append(entry)
    return method, precursors


def run(cfg):
    with open("input_structures.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    try:
        llm = CSLLM(cfg["weights_dir"], cfg["device"], cfg["dtype"])
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - missing weights/libraries -> report, don't crash
        print(f"[synthesizability] CSLLM unavailable: {exc}")
        llm, status = None, "unavailable"

    results = []
    by_formula = {}
    tols = [cfg["symprec"], 0.05, 0.1]
    for item in payload if llm else []:
        structure = Structure.from_dict(item["structure"])
        formula = structure.composition.reduced_formula
        result = {"uuid": item["uuid"], "formula": formula, "material_string": None,
                  "spacegroup": None, "synthesizability": None, "method": None,
                  "precursors": None, "notes": []}
        try:
            text, spg, tol_used = material_string_with_fallback(structure, tols)
            result.update({"material_string": text, "spacegroup": spg, "symmetry_tol": tol_used})
            probs, mass = label_probabilities(
                llm.sequence_probs("synthesis", SYN_PROMPT.format(text), SYN_LABELS))
            p_true = probs["True"]
            result["synthesizability"] = {
                "p_true": round(p_true, 4),
                "label_mass": round(mass, 4),
                "entropy_bits": round(binary_entropy_bits(p_true), 4),
            }
        except Exception as exc:  # noqa: BLE001 - one bad structure must not sink the batch
            result["notes"].append(f"synthesizability failed: {exc}")

        try:
            if formula not in by_formula:
                by_formula[formula] = predict_formula(llm, formula, cfg)
            result["method"], result["precursors"] = by_formula[formula]
        except Exception as exc:  # noqa: BLE001
            result["notes"].append(f"method/precursor prediction failed: {exc}")
        results.append(result)

    output = {
        "results": results,
        "config": {
            "weights_dir": cfg["weights_dir"],
            "adapters": _ADAPTERS,
            "base": _BASE,
            "device": cfg["device"],
            "dtype": getattr(llm, "dtype", cfg["dtype"]),
            "predict_method": cfg["predict_method"],
            "predict_precursors": cfg["predict_precursors"],
            "precursor_num_sets": cfg["precursor_num_sets"],
            "symprec": cfg["symprec"],
        },
        "status": status,
    }
    with open("output.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh)


def _bool(text):
    return str(text).strip().lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto")   # auto | bfloat16 | float16 | float32
    parser.add_argument("--predict_method", type=_bool, default=True)
    parser.add_argument("--predict_precursors", type=_bool, default=True)
    parser.add_argument("--precursor_num_sets", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--symprec", type=float, default=0.01)
    args = parser.parse_args()
    run(vars(args))

"""
For every input structure it predicts:
* the fundamental band gap, from one or more pretrained ML property models
  (matgl MEGNet multi-fidelity -- the workhorse; ALIGNN if importable);
* absolute band-edge positions (CBM / VBM) on the RHE and vacuum scales via the
  empirical Butler--Ginley / Mulliken-electronegativity relation

      E_CB(V vs NHE) = chi - E_e - 0.5 * E_g ,   E_VB = E_CB + E_g

  with ``E_e = 4.5`` eV and ``chi`` the stoichiometry-weighted geometric-mean
  Mulliken electronegativity of the constituents (chi_i = (IE_i + EA_i)/2,
  from pymatgen). At pH 0 the NHE and RHE scales coincide and, because oxide
  edges shift ~Nernstian, the RHE-scale numbers are ~pH-independent -- which is
  the scale the downstream photocatalytic straddle test uses;
* a coarse visible-light absorption label from the gap.

The photocatalytic *straddle* verdict (does the gap bracket a reaction's redox
couple with margin?) is NOT computed here -- it needs the uvsib reaction
modules, which this bare remote script does not import. ``OpticalScreenWorkChain``
adds ``band_info["straddle"]`` after this job returns.

Input (``input_structures.json``, staged via the ``file`` namespace):

    [{"uuid": <bulk structure uuid>, "structure": <pymatgen Structure.as_dict()>}, ...]

CLI:

    --models=megnet_mfi,alignn_mbj   comma list; unknown / failed models are skipped
    --megnet_fidelity=2              matgl mfi index (0 PBE, 1 GLLB-SC, 2 HSE, 3 SCAN)
    --gap_min=1.4 --gap_max=3.1      visible-light window (eV) -- label only
    --pH=0.0                         pH the RHE-scale edges are reported at

Output (``output.json``, parsed into ``output_dict`` by ``electronic_parser``):

    {"results": [{"uuid": ..., "band_info": { ... see build_band_info() ... }}, ...],
     "config":  {"models_used": [...], "megnet_fidelity": 2, "pH": 0.0, ...},
     "status":  "ok" | "unavailable"}        # "unavailable": no gap model importable
"""

import argparse
import json
import math

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

E_E_EV = 4.5  # free-electron energy vs NHE used by the Butler--Ginley relation

_MEGNET_MFI = {0: "PBE", 1: "GLLB-SC", 2: "HSE", 3: "SCAN"}

# lazily loaded, then reused across all input structures in the job
_MEGNET_MODEL = None
_MEGNET_MODEL_NAME = None

# matgl >= 2 (PyG backend) renamed the model; the 1.x name is kept as a fallback.
# matgl 1.1.3 (DGL) gives unphysical gaps with this model (e.g. MgO ~0.3 eV at
# HSE fidelity) -- see docs/venv_electronic_build.md -- so use matgl >= 4.
_MEGNET_NAMES = ("MEGNet-BandGap-mfi-MP-2019.4.1", "MEGNet-MP-2019.4.1-BandGap-mfi")


# --------------------------------------------------------------------------- #
# ML band-gap models
# --------------------------------------------------------------------------- #
def _load_megnet():
    global _MEGNET_MODEL, _MEGNET_MODEL_NAME
    if _MEGNET_MODEL is None:
        import matgl
        errors = []
        for name in _MEGNET_NAMES:
            try:
                _MEGNET_MODEL = matgl.load_model(name)
                _MEGNET_MODEL_NAME = f"{name} (matgl {matgl.__version__})"
                break
            except Exception as exc:  # noqa: BLE001 - unknown name on this matgl version
                errors.append(f"{name}: {exc}")
        else:
            raise RuntimeError("no MEGNet band-gap model loadable: " + "; ".join(errors))
    return _MEGNET_MODEL


def megnet_gap(structure, fidelity):
    """MEGNet multi-fidelity gap (eV) at the requested fidelity index. The
    ``predict_structure`` state kwarg was renamed across matgl releases, so try
    both spellings before the positional fallback."""
    import torch

    model = _load_megnet()
    attr = torch.tensor([int(fidelity)])
    for kw in ("state_attr", "state_feats"):
        try:
            return float(model.predict_structure(structure=structure, **{kw: attr}))
        except TypeError:
            continue
    return float(model.predict_structure(structure, attr))


def alignn_gap(structure, model_name="mp_gappbe_alignn"):
    """ALIGNN pretrained gap (eV). Optional cross-check -- the ALIGNN
    ``pretrained`` API is version-dependent, so any failure just drops this
    model from the ensemble."""
    from alignn.pretrained import get_prediction
    from jarvis.core.atoms import pmg_to_atoms

    pred = get_prediction(model_name=model_name, atoms=pmg_to_atoms(structure))
    # get_prediction returns a scalar on some ALIGNN releases and a 1-element
    # list/array on others (e.g. alignn 2024.5.27) -- flatten to a float either way.
    while isinstance(pred, (list, tuple)):
        pred = pred[0]
    return float(pred)


_GAP_MODELS = {
    "megnet_mfi": ("MEGNet multi-fidelity", lambda s, cfg: megnet_gap(s, cfg["megnet_fidelity"])),
    "alignn_pbe": ("ALIGNN JARVIS PBE",     lambda s, cfg: alignn_gap(s, "mp_gappbe_alignn")),
    "alignn_mbj": ("ALIGNN JARVIS MBJ",     lambda s, cfg: alignn_gap(s, "jv_mbj_bandgap_alignn")),
}


# --------------------------------------------------------------------------- #
# Butler--Ginley / Mulliken band edges
# --------------------------------------------------------------------------- #
def mulliken_electronegativity(structure):
    """Stoichiometry-weighted geometric-mean absolute (Mulliken)
    electronegativity of the constituents, in eV: chi_i = (IE_i + EA_i)/2,
    both from pymatgen. Elements with no tabulated electron affinity are
    treated as EA = 0."""
    log_sum = 0.0
    n_sum = 0.0
    for site in structure:
        for sp, amt in site.species.get_el_amt_dict().items():
            el = Element(sp)
            ie = el.ionization_energy
            if ie is None:
                raise ValueError(f"no ionization energy tabulated for {sp}")
            ea = el.electron_affinity
            ea = float(ea) if ea is not None else 0.0
            chi_i = 0.5 * (float(ie) + ea)
            log_sum += amt * math.log(chi_i)
            n_sum += amt
    return math.exp(log_sum / n_sum)


def build_band_info(structure, gap_values, cfg):
    """Assemble the ``band_info`` payload for one structure. ``gap_values`` is
    ``{model_key: gap_eV}`` for every model that succeeded (possibly empty)."""
    notes = []
    gap_mean = gap_std = cb = vb = chi = None
    edges_rhe = edges_vac = None

    if gap_values:
        vals = list(gap_values.values())
        gap_mean = sum(vals) / len(vals)
        gap_std = (sum((v - gap_mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        try:
            chi = mulliken_electronegativity(structure)
            cb = chi - E_E_EV - 0.5 * gap_mean
            vb = cb + gap_mean
            edges_rhe = {"cb": round(cb, 4), "vb": round(vb, 4), "pH": cfg["pH"],
                         "method": "butler-ginley/mulliken", "E_e_eV": E_E_EV}
            edges_vac = {"cb": round(-(cb + E_E_EV), 4), "vb": round(-(vb + E_E_EV), 4)}
        except ValueError as exc:
            notes.append(f"no band edges: {exc}")
    else:
        notes.append("no ML gap model importable on the Electronic code environment")

    gap_min, gap_max = cfg["gap_min"], cfg["gap_max"]
    if gap_mean is None:
        absorption = None
    else:
        if gap_mean < 0.3:
            regime = "metallic"
        elif gap_mean < gap_min:
            regime = "narrow-gap"
        elif gap_mean <= gap_max:
            regime = "visible"
        else:
            regime = "uv"
        absorption = {
            "regime": regime,
            "absorbs_visible": bool(gap_min <= gap_mean <= gap_max),
            "onset_nm": round(1239.84 / gap_mean, 1) if gap_mean > 0.05 else None,
            "window_eV": [gap_min, gap_max],
        }

    return {
        "screen": "ml_no_dft",
        "screen_version": 1,
        "gap_eV": round(gap_mean, 4) if gap_mean is not None else None,
        "gap_std_eV": round(gap_std, 4) if gap_std is not None else None,
        "gap_values_eV": {k: round(v, 4) for k, v in gap_values.items()},
        "gap_models": list(gap_values),
        "megnet_fidelity": cfg["megnet_fidelity"],
        "megnet_fidelity_label": _MEGNET_MFI.get(cfg["megnet_fidelity"]),
        "direct_gap": None,
        "mulliken_electronegativity_eV": round(chi, 4) if chi is not None else None,
        "band_edges_vs_rhe_V": edges_rhe,
        "band_edges_vs_vacuum_eV": edges_vac,
        "absorption": absorption,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(models, cfg):
    with open("input_structures.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    usable = []
    for key in models:
        if key not in _GAP_MODELS:
            continue
        try:  # import-check once, up front
            if key == "megnet_mfi":
                _load_megnet()
            else:
                import alignn.pretrained  # noqa: F401
                import jarvis.core.atoms  # noqa: F401
            usable.append(key)
        except Exception as exc:  # noqa: BLE001 - any import/load failure drops the model
            print(f"[electronic] model {key!r} unavailable: {exc}")

    results = []
    for item in payload:
        structure = Structure.from_dict(item["structure"])
        gap_values = {}
        for key in usable:
            try:
                gap_values[key] = _GAP_MODELS[key][1](structure, cfg)
            except Exception as exc:  # noqa: BLE001 - skip this model for this structure
                print(f"[electronic] {key} failed on {item['uuid']}: {exc}")
        results.append({"uuid": item["uuid"], "band_info": build_band_info(structure, gap_values, cfg)})

    output = {
        "results": results,
        "config": {
            "models_requested": models,
            "models_used": usable,
            "megnet_fidelity": cfg["megnet_fidelity"],
            "megnet_model": _MEGNET_MODEL_NAME,
            "pH": cfg["pH"],
            "gap_window_eV": [cfg["gap_min"], cfg["gap_max"]],
        },
        "status": "ok" if usable else "unavailable",
    }
    with open("output.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="megnet_mfi")
    parser.add_argument("--megnet_fidelity", type=int, default=2)
    parser.add_argument("--gap_min", type=float, default=1.4)
    parser.add_argument("--gap_max", type=float, default=3.1)
    parser.add_argument("--pH", type=float, default=0.0)
    args = parser.parse_args()

    run(
        [m.strip() for m in args.models.split(",") if m.strip()],
        {
            "megnet_fidelity": args.megnet_fidelity,
            "gap_min": args.gap_min,
            "gap_max": args.gap_max,
            "pH": args.pH,
        },
    )

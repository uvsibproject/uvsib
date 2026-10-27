"""Target redox couples for the photocatalytic band-edge *straddle* test.

One place that maps a ``(reaction, reaction_path)`` -- the SAME normalized
strings used as ``DBSurfaceMLAdsorbate.reaction`` / ``.reaction_path`` and as
``step_status`` keys -- to the pair of standard potentials the photo-generated
carriers must bracket:

    u_red : reduction-couple potential (V vs RHE); the CB electron must sit
            *above* it  (i.e. E_CB <= u_red on the electrochemical axis)
    u_ox  : oxidation-couple potential (V vs RHE); the VB hole must sit
            *below* it  (i.e. E_VB >= u_ox)

For a solar-fuel material the fuel-forming half-reaction is the one under test
and the partner is almost always water oxidation (``u_ox = 1.23``). OER / CER
are the oxidation photo-anode case: the material does the oxidation and the
partner is hydrogen evolution (``u_red = 0``).

The per-pathway potentials come from the reaction modules' own
``equilibrium_potential`` entries (single source of truth, imported lazily) so
this can never drift from the CHE overpotential calculators. Importing this
module therefore transitively loads ``uvsib.workflows.settings`` (AiiDA
profile) -- call it from a profile-loaded context, like
``uvsib.workchains.pipeline_report.step_labels`` already requires.
"""

import importlib

WATER_OX_RHE = 1.23  # O2 / H2O, 4e-  (V vs RHE)
HER_RED_RHE = 0.0    # H+ / H2         (V vs RHE)

# reaction -> (module, *_PATHWAYS dict name, role of the *material* in a
# solar-fuel cell: it drives this half-reaction, the partner is the other one)
_REACTION_MODULES = {
    "HER": ("her", "HER_PATHWAYS", "reduction"),
    "ORR": ("orr", "ORR_PATHWAYS", "reduction"),
    "CO2RR": ("co2rr", "CO2RR_PATHWAYS", "reduction"),
    "NRR": ("nrr", "NRR_PATHWAYS", "reduction"),
    "NOXRR": ("noxrr", "NOXRR_PATHWAYS", "reduction"),
    "CER": ("cer", "CER_PATHWAYS", "oxidation"),
}


def _pathway_potentials(mod_name, dict_name):
    module = importlib.import_module(f"uvsib.workchains.{mod_name}")
    pathways = getattr(module, dict_name)
    return {path: float(spec["equilibrium_potential"]) for path, spec in pathways.items()}


def _couple(role, u_red, u_ox, label):
    return {"role": role, "u_red": round(float(u_red), 4), "u_ox": round(float(u_ox), 4), "label": label}


def all_couples():
    """``{reaction: {reaction_path: {"role", "u_red", "u_ox", "label"}}}`` for
    every implemented (reaction, pathway). OER has no ``*_PATHWAYS`` dict, so its
    single route is keyed ``"default"`` (matching ``check_valid`` in
    ``workflows/workflows.py``)."""
    couples = {
        "OER": {"default": _couple("oxidation", HER_RED_RHE, WATER_OX_RHE, "O2/H2O  vs  H+/H2")},
    }
    for reaction, (mod, dname, role) in _REACTION_MODULES.items():
        couples[reaction] = {}
        for path, u_eq in _pathway_potentials(mod, dname).items():
            if role == "reduction":
                couples[reaction][path] = _couple(
                    "reduction", u_eq, WATER_OX_RHE,
                    f"{reaction}:{path} (u_red={u_eq:+.2f} V)  vs  O2/H2O")
            else:  # oxidation photo-anode (CER): equilibrium_potential is u_ox
                couples[reaction][path] = _couple(
                    "oxidation", HER_RED_RHE, u_eq,
                    f"{reaction}:{path} (u_ox={u_eq:+.2f} V)  vs  H+/H2")
    return couples


def couple_for(reaction, reaction_path):
    """The couple for one ``(reaction, reaction_path)``, tolerant of the
    OER ``default`` / ``none`` / ``""`` spellings the frontend may pass.
    Returns ``None`` for an unknown reaction/path."""
    reaction = (reaction or "").strip().upper()
    by_path = all_couples().get(reaction, {})
    if not by_path:
        return None
    rp = (reaction_path or "").strip().lower()
    if rp in by_path:
        return by_path[rp]
    if len(by_path) == 1:                    # OER, or any single-route reaction
        return next(iter(by_path.values()))
    return by_path.get("default")


def straddle_verdict(cb_vs_rhe, vb_vs_rhe, u_red, u_ox, margin=0.2):
    """Does a gap with edges ``(cb, vb)`` (V vs RHE) bracket ``[u_red, u_ox]``
    with at least ``margin`` volts of headroom on each side?"""
    margin_red = round(u_red - cb_vs_rhe, 4)   # want >= margin  (CB above the reduction level)
    margin_ox = round(vb_vs_rhe - u_ox, 4)     # want >= margin  (VB below the oxidation level)
    margin = round(float(margin), 4)
    return {
        "u_red": round(float(u_red), 4),
        "u_ox": round(float(u_ox), 4),
        "margin_required_V": margin,
        "min_gap_eV": round((u_ox - u_red) + 2 * margin, 4),
        "margin_reduction_V": margin_red,
        "margin_oxidation_V": margin_ox,
        "straddles": bool(margin_red >= margin and margin_ox >= margin),
    }


def model_band_edges(edges_rhe, gap_values):
    """``{model_key: {"cb", "vb"}}`` (V vs RHE): each gap model's own band edges.
    Butler--Ginley puts every gap symmetrically around the composition-only
    mid-gap level chi - E_e, so it is taken from the mean-gap ``edges_rhe`` and
    each model's gap (negative ones clamped to 0) is centred on it."""
    mid = 0.5 * (edges_rhe["cb"] + edges_rhe["vb"])
    return {key: {"cb": round(mid - 0.5 * max(gap, 0.0), 4), "vb": round(mid + 0.5 * max(gap, 0.0), 4)}
            for key, gap in (gap_values or {}).items()}

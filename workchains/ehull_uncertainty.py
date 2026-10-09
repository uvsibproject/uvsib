"""E_above_hull committee uncertainty -- pure analysis (no AiiDA / DB access).

Given, for the primary MLIP and every committee MLIP, the energies of the SAME
structures (committee energies are single points on the primary-relaxed
geometries), build one phase diagram per model and report how much the target
candidates' stability changes across models. Used by EhullUncertaintyWorkChain;
kept free of AiiDA / database imports so it can be unit-tested directly.

Rules
-----
* Every model's hull is built from the same structures: the intersection of the
  uuids every model has an energy for. A phase missing from one model's hull
  would bias that model's E_hull low, so it is dropped from all of them (and
  reported in ``dropped``).
* The spread is taken over the SIGNED distance ``ed`` of the entry to the hull
  of ALL OTHER structures (only the entry itself removed, polymorphs kept):
  positive = E_hull for an entry above the hull, negative = the margin by which
  a ground state beats its best competitor (another polymorph or a
  decomposition). E_hull itself is clipped at 0, which would hide disagreement
  among models that all call a phase stable. pymatgen's
  ``get_decomp_and_phase_separation_energy`` is NOT used: it drops every
  polymorph of the composition, so a metastable polymorph would get a large
  negative value and ``ed`` would jump whenever the models disagree on the
  ground-state polymorph. E_hull is still reported (it is what the selection
  used).
* Geometry disagreement: the committee energies are single points on the
  primary geometries, which are not exactly the committee model's minima. For
  each structure and committee model the energy that model would gain by
  relaxing is estimated harmonically from the single-point forces and stress
  (``relaxation_energy``):

      dE_relax = <|F_i|^2> / (2 k)  +  V_atom |sigma_dev|^2 / (4 G)    [per atom]

  with an effective force constant ``k`` (eV/A^2) and shear modulus ``G``
  (GPa). Only the DEVIATORIC stress enters: a uniform pressure is mostly a
  systematic lattice-constant offset of the model, which largely cancels in
  E_hull. A model's signed hull distance can move by at most
  ``delta_m = dE_relax(target) + sum_j x_j dE_relax(j)`` over the phases j (atom
  fractions x_j) it is compared against. The target is flagged when the largest
  ``delta_m`` exceeds ``max(std(ed), relax_energy_floor)``: the possible
  geometry shift is as large as the model disagreement itself (and not
  negligible), so the spread cannot be trusted on its own. The structure that
  contributes most to that ``delta_m`` is reported as the source -- the target
  itself (``self``) or a competing phase (``competing``).
"""

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram

_EPS = 1e-8
GPA_A3_TO_EV = 6.241509074e-3  # 1 GPa * 1 A^3 in eV


def _stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def deviatoric_stress_sq(stress_voigt):
    """``|sigma_dev|^2`` (GPa^2, Frobenius) from a Voigt stress
    ``[xx, yy, zz, yz, xz, xy]``."""
    s = np.asarray(stress_voigt, dtype=float)
    diag = s[:3] - s[:3].mean()
    return float(np.sum(diag ** 2) + 2.0 * np.sum(s[3:] ** 2))


def relaxation_energy(diag, volume_per_atom, force_constant, shear_modulus):
    """Harmonic estimate (eV/atom) of the energy a model gains by relaxing a
    structure from the geometry its single point was taken on. Returns
    ``(total, force_part, stress_part)``; ``None`` parts if data is missing."""
    if not diag:
        return None, None, None
    force_sq = diag.get("force_sq_mean")
    e_force = force_sq / (2.0 * force_constant) if force_sq is not None else None
    stress = diag.get("stress_voigt")
    e_stress = (volume_per_atom * deviatoric_stress_sq(stress) / (4.0 * shear_modulus) * GPA_A3_TO_EV
                if stress is not None else None)
    if e_force is None and e_stress is None:
        return None, None, None
    return (e_force or 0.0) + (e_stress or 0.0), e_force, e_stress


def signed_hull_distance(pd, entry, tol=_EPS):
    """``(decomposition, ed)`` of ``entry`` (a member of ``pd``) relative to the
    hull of all other entries: ``ed`` = E_hull > 0 above the hull, <= 0 for a
    hull entry (margin to its best competitor)."""
    decomp, ehull = pd.get_decomp_and_e_above_hull(entry)
    if ehull > tol:
        return decomp, float(ehull)
    others = [e for e in pd.all_entries if e is not entry]
    decomp, ed = PhaseDiagram(others).get_decomp_and_e_above_hull(entry, allow_negative=True)
    return decomp, float(ed)


def committee_ehull(target_uuids, entries, diagnostics, primary,
                    ehull_threshold, force_constant, shear_modulus,
                    relax_energy_floor):
    """Per-model E_hull of ``target_uuids`` and their committee spread.

    Parameters
    ----------
    target_uuids : list[str]
        Selected candidates of the target formula (``ml_selection`` uuids).
    entries : dict[str, dict[str, ComputedStructureEntry]]
        ``{method: {uuid: entry}}`` for the primary and every committee model,
        elemental references included. Each entry's ``data["struct_uuid"]``
        must be its uuid.
    diagnostics : dict[str, dict[str, dict]]
        ``{committee_method: {uuid: {"max_force", "force_sq_mean",
        "stress_voigt", "max_stress", "pressure"}}}`` from the single points
        (forces eV/A, eV^2/A^2; stresses GPa).
    primary : str
        Method of the primary MLIP (a key of ``entries``).
    ehull_threshold : float
        E_hull (eV/atom) under which a model "votes" the candidate stable.
    force_constant : float
        Effective force constant k (eV/A^2) of the harmonic relaxation estimate.
    shear_modulus : float
        Shear modulus G (GPa) of the harmonic relaxation estimate.
    relax_energy_floor : float
        Geometry shifts below this (eV/atom) never flag a target.

    Returns
    -------
    dict (JSON-serialisable)
        ``{"methods", "n_common", "dropped", "missing_targets", "per_uuid"}``.

    Raises
    ------
    ValueError
        From PhaseDiagram if the common structure set lacks an elemental
        terminal for some model.
    """
    committee = [m for m in entries if m != primary]
    methods = [primary] + committee

    uuid_sets = [set(entries[m]) for m in methods]
    common = set.intersection(*uuid_sets)
    dropped = sorted(set.union(*uuid_sets) - common)

    pds = {m: PhaseDiagram([entries[m][u] for u in sorted(common)]) for m in methods}

    relax = {}
    for m in committee:
        relax[m] = {}
        for u in common:
            structure = entries[m][u].structure
            relax[m][u] = relaxation_energy(diagnostics.get(m, {}).get(u),
                                            structure.volume / len(structure),
                                            force_constant, shear_modulus)

    per_uuid = {}
    missing_targets = []
    for uuid in target_uuids:
        if uuid not in common:
            missing_targets.append(uuid)
            continue

        ehull, ed, ef, decomp_uuids, decomp_fractions = {}, {}, {}, {}, {}
        for m in methods:
            pd = pds[m]
            entry = entries[m][uuid]
            ehull[m] = float(pd.get_e_above_hull(entry))
            decomp, ed[m] = signed_hull_distance(pd, entry)
            ef[m] = float(pd.get_form_energy_per_atom(entry))
            decomp_fractions[m] = {}
            for d, x in (decomp or {}).items():
                du = str(d.data["struct_uuid"])
                decomp_fractions[m][du] = decomp_fractions[m].get(du, 0.0) + float(x)
            decomp_uuids[m] = sorted(decomp_fractions[m])

        ed_stats = _stats([ed[m] for m in methods])
        ehull_stats = _stats([ehull[m] for m in methods])
        committee_ed = [ed[m] for m in committee]
        votes = sum(1 for m in methods if ehull[m] <= ehull_threshold + _EPS)

        own = {m: diagnostics.get(m, {}).get(uuid) for m in committee}
        forces = [d["max_force"] for d in own.values() if d and d.get("max_force") is not None]
        stresses = [d["max_stress"] for d in own.values() if d and d.get("max_stress") is not None]

        # possible geometry shift of each committee model's ed: the target's
        # own relaxation energy + the fraction-weighted ones of the phases it
        # is compared against
        geometry_shift, contributions = {}, {}
        for m in committee:
            parts = [(uuid, 1.0, relax[m][uuid][0])]
            parts += [(u, x, relax[m][u][0]) for u, x in decomp_fractions[m].items() if u != uuid]
            contributions[m] = sorted(
                [{"uuid": u, "model": m, "fraction": x, "relax_energy": e,
                  "shift": x * e, "self": u == uuid,
                  "max_force": (diagnostics.get(m, {}).get(u) or {}).get("max_force"),
                  "max_stress": (diagnostics.get(m, {}).get(u) or {}).get("max_stress"),
                  "pressure": (diagnostics.get(m, {}).get(u) or {}).get("pressure")}
                 for u, x, e in parts if e is not None],
                key=lambda c: -c["shift"])
            geometry_shift[m] = float(sum(c["shift"] for c in contributions[m]))

        worst_model = max(geometry_shift, key=geometry_shift.get) if geometry_shift else None
        shift_max = geometry_shift[worst_model] if worst_model else 0.0
        geometry_limit = max(ed_stats["std"], relax_energy_floor)
        flagged = shift_max > geometry_limit
        dominant = contributions[worst_model][0] if worst_model and contributions[worst_model] else None
        culprits = [c for c in contributions[worst_model] if c["shift"] > 0] if flagged else []

        if flagged:
            label = "geometry_disagreement"
        elif votes == len(methods):
            label = "robust"
        elif votes == 0:
            label = "unstable"
        else:
            label = "uncertain"

        per_uuid[uuid] = {
            "ehull": ehull,
            "ed": ed,
            "ef": ef,
            "decomposition": decomp_uuids,
            "ed_mean": ed_stats["mean"],
            "ed_std": ed_stats["std"],
            "ed_min": ed_stats["min"],
            "ed_max": ed_stats["max"],
            "ehull_mean": ehull_stats["mean"],
            "ehull_std": ehull_stats["std"],
            "ehull_min": ehull_stats["min"],
            "ehull_max": ehull_stats["max"],
            "bias_primary": float(ed[primary] - np.mean(committee_ed)) if committee_ed else 0.0,
            "stable_votes": votes,
            "n_models": len(methods),
            "max_force": {m: (d or {}).get("max_force") for m, d in own.items()},
            "max_stress": {m: (d or {}).get("max_stress") for m, d in own.items()},
            "pressure": {m: (d or {}).get("pressure") for m, d in own.items()},
            "worst_force": max(forces) if forces else None,
            "worst_stress": max(stresses) if stresses else None,
            "relax_energy": {m: relax[m][uuid][0] for m in committee},
            "geometry_shift": geometry_shift,
            "geometry_shift_max": shift_max,
            "geometry_shift_model": worst_model,
            "geometry_limit": geometry_limit,
            "geometry_source": (("self" if dominant["self"] else "competing")
                                if dominant else None),
            "geometry_via": ({"uuid": dominant["uuid"], "model": dominant["model"],
                              "shift": dominant["shift"]} if dominant else None),
            "geometry_disagreement": flagged,
            "geometry_culprits": culprits,
            "label": label,
        }

    return {
        "methods": methods,
        "n_common": len(common),
        "dropped": dropped,
        "missing_targets": missing_targets,
        "per_uuid": per_uuid,
    }

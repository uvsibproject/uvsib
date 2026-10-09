"""Surface-energy and adsorption / overpotential committee uncertainty -- pure
analysis (no AiiDA / DB access).

The primary MLIP (``face_build.model`` / ``adsorbates.model``) relaxed the slabs
and adsorbate systems; committee MLIPs re-score the SAME geometries by single
points. Every derived quantity is computed entirely within one model, then the
spread across models is reported. Used by SurfaceUncertaintyWorkChain and
AdsorptionUncertaintyWorkChain; kept free of AiiDA / database imports so it can
be unit-tested directly.

Surface energy (exactly as ``codes/files/slab_relax.py``)::

    gamma_m = (E_slab,m - N_slab * eps_bulk,m) / (2 A)

with eps_bulk,m the model's energy per atom of the bulk the slab was cut from
(on the same bulk geometry the primary's epa used).

Overpotential: the model's energies (clean slab ``*``, intermediates, gas
references per molecule) go through the SAME reaction function the pipeline
uses (``calculate_<reaction>_overpotential``), so ZPE, CHE bookkeeping and any
thermodynamic pin are identical for every model.

Geometry check (forces only -- the slab cell is never relaxed, so stress plays
no role; energies are TOTAL, since gamma and Delta G are not per-atom
quantities). The energy model m would gain by relaxing a structure is
estimated harmonically, ``dE_relax = sum_i |F_i|^2 / (2 k)``:

* surface:  dgamma_m = [dE_relax(slab) + N_slab * <|F|^2>_bulk / (2 k)] / (2 A)
* adsorption: each step's Delta G is linear in the species energies,
  dG_i = sum_s c_is E_s (coefficients obtained by finite differences through
  the reaction function); its possible shift is sum_s |c_is| dE_relax(s), and
  deta_m = max_i of that (eta follows the largest one-electron step).

A slab / candidate is flagged ("geometry") when max_m of the shift exceeds
max(spread, floor): the possible geometry effect is as large as the model
disagreement itself, so the spread cannot be trusted on its own.
"""

from itertools import combinations
import numpy as np

_FD_STEP = 1e-3  # eV, finite-difference step for the dG_i(E_s) coefficients


def _stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _kendall_tau(order_a, order_b):
    """Kendall tau between two rankings of the same items (lists of ids,
    best first). ``None`` for fewer than two common items."""
    common = [i for i in order_a if i in order_b]
    if len(common) < 2:
        return None
    pos_b = {item: k for k, item in enumerate(order_b)}
    concordant = discordant = 0
    for x, y in combinations(common, 2):
        # x precedes y in order_a
        if pos_b[x] < pos_b[y]:
            concordant += 1
        else:
            discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def _ranking(values):
    """Ids sorted by ascending value."""
    return [k for k, _ in sorted(values.items(), key=lambda kv: kv[1])]


# --------------------------------------------------------------------------- #
# surface energy
# --------------------------------------------------------------------------- #
def committee_surfaces(slabs, primary, force_constant, gamma_floor, n_selected):
    """Per-slab gamma spread + per-bulk facet-ranking agreement.

    Parameters
    ----------
    slabs : list[dict]
        One dict per slab: ``{"surface_id", "bulk_uuid", "area" (A^2),
        "n_atoms", "gamma": {model: eV/A^2} (primary + committee),
        "diag": {committee_model: {"slab_force_sq_sum", "bulk_force_sq_mean",
        "slab_max_force", "bulk_max_force"}}}``. Only models present in
        ``gamma`` of EVERY slab of a bulk are used for that bulk's ranking.
    primary : str
        Primary model name (a key of every ``gamma``).
    force_constant : float
        k (eV/A^2) of the harmonic relaxation estimate.
    gamma_floor : float
        Geometry shifts (eV/A^2) below this never flag a slab.
    n_selected : int
        Facets per bulk carried on to adsorption (MAX_NUM_ADS): reports
        whether every model would select the same ones.

    Returns
    -------
    dict (JSON-serialisable): ``{"per_slab": {surface_id: {...}},
    "per_bulk": {bulk_uuid: {...}}}``.
    """
    per_slab = {}
    by_bulk = {}
    for slab in slabs:
        gamma = slab["gamma"]
        methods = [primary] + sorted(m for m in gamma if m != primary)
        committee = methods[1:]
        st = _stats([gamma[m] for m in methods])
        area, n_atoms = slab["area"], slab["n_atoms"]

        shift, parts = {}, {}
        for m in committee:
            d = slab.get("diag", {}).get(m) or {}
            e_slab = (d.get("slab_force_sq_sum") or 0.0) / (2.0 * force_constant)
            e_bulk = n_atoms * (d.get("bulk_force_sq_mean") or 0.0) / (2.0 * force_constant)
            parts[m] = {"slab": e_slab / (2.0 * area), "bulk": e_bulk / (2.0 * area)}
            shift[m] = parts[m]["slab"] + parts[m]["bulk"]
        worst = max(shift, key=shift.get) if shift else None
        shift_max = shift[worst] if worst else 0.0
        limit = max(st["std"], gamma_floor)
        flagged = shift_max > limit
        source = None
        if worst:
            source = "slab" if parts[worst]["slab"] >= parts[worst]["bulk"] else "bulk"

        per_slab[slab["surface_id"]] = {
            "gamma": dict(gamma),
            "gamma_mean": st["mean"], "gamma_std": st["std"],
            "gamma_min": st["min"], "gamma_max": st["max"],
            "bias_primary": (float(gamma[primary] - np.mean([gamma[m] for m in committee]))
                             if committee else 0.0),
            "n_models": len(methods),
            "max_force": {m: (slab.get("diag", {}).get(m) or {}).get("slab_max_force")
                          for m in committee},
            "geometry_shift": shift,
            "geometry_shift_parts": parts,
            "geometry_shift_max": shift_max,
            "geometry_shift_model": worst,
            "geometry_limit": limit,
            "geometry_source": source,
            "geometry_disagreement": flagged,
            # filled in below, per bulk
            "rank": {},
            "rank_agree": None,
            "label": None,
        }
        by_bulk.setdefault(slab["bulk_uuid"], []).append(slab)

    per_bulk = {}
    for bulk_uuid, bulk_slabs in by_bulk.items():
        models = set.intersection(*(set(s["gamma"]) for s in bulk_slabs))
        methods = [primary] + sorted(m for m in models if m != primary)
        rankings = {m: _ranking({s["surface_id"]: s["gamma"][m] for s in bulk_slabs})
                    for m in methods}
        for s in bulk_slabs:
            res = per_slab[s["surface_id"]]
            res["rank"] = {m: rankings[m].index(s["surface_id"]) + 1 for m in methods}
            res["rank_agree"] = len(set(res["rank"].values())) == 1
            if res["geometry_disagreement"]:
                res["label"] = "geometry_disagreement"
            else:
                res["label"] = "robust" if res["rank_agree"] else "uncertain"
        top1 = {m: rankings[m][0] for m in methods}
        selected = {m: sorted(rankings[m][:n_selected]) for m in methods}
        per_bulk[bulk_uuid] = {
            "methods": methods,
            "n_slabs": len(bulk_slabs),
            "top1": top1,
            "top1_agree": len(set(top1.values())) == 1,
            "selected": selected,
            "selected_agree": len({tuple(v) for v in selected.values()}) == 1,
            "kendall_tau": {m: _kendall_tau(rankings[primary], rankings[m])
                            for m in methods if m != primary},
        }
    return {"per_slab": per_slab, "per_bulk": per_bulk}


# --------------------------------------------------------------------------- #
# adsorption / overpotential
# --------------------------------------------------------------------------- #
def _species_source(name):
    if name == "*":
        return "slab"
    return "adsorbate" if str(name).startswith("*") else "gas"


def step_coefficients(calc_fn, energy_set, pathway, h=_FD_STEP):
    """``{species: [c_i, ...]}``: d(dG_i)/d(E_species) of every step through
    ``calc_fn`` (linear in the energies, so finite differences are exact up to
    round-off)."""
    _, base, _ = calc_fn(dict(energy_set), pathway)
    coeffs = {}
    for species in energy_set:
        shifted = dict(energy_set)
        shifted[species] = shifted[species] + h
        _, dg, _ = calc_fn(shifted, pathway)
        coeffs[species] = [(a - b) / h for a, b in zip(dg, base)]
    return coeffs


def committee_adsorption(candidates, primary, calc_fn, pathway, force_constant,
                         eta_tolerance, eta_floor):
    """Per-candidate eta spread + per-bulk best-candidate agreement.

    Parameters
    ----------
    candidates : list[dict]
        One dict per DBSurfaceMLAdsorbate row: ``{"row_id", "bulk_uuid",
        "surface_id", "energies": {model: {species: E}} (primary + committee;
        gas references per molecule), "force_sq_sum": {committee_model:
        {species: sum |F|^2 (per molecule for gas)}}, "max_force_ads":
        {committee_model: {species: eV/A}}}``.
    primary : str
        Primary model name.
    calc_fn : callable
        The reaction's ``calculate_<reaction>_overpotential(energy_set, path)``
        returning ``(eta, dG_steps, dG_cumulative)``.
    pathway : str
        Reaction path passed to ``calc_fn``.
    force_constant : float
        k (eV/A^2) of the harmonic relaxation estimate.
    eta_tolerance : float
        std(eta) (V) above which a candidate is "uncertain".
    eta_floor : float
        Geometry shifts (V) below this never flag a candidate.

    Returns
    -------
    dict (JSON-serialisable): ``{"per_candidate": {row_id: {...}},
    "per_bulk": {bulk_uuid: {...}}, "failed": [{row_id, model, reason}]}``.
    """
    per_candidate = {}
    failed = []
    for cand in candidates:
        results = {}
        for m, energy_set in cand["energies"].items():
            try:
                eta, dg_steps, dg_cum = calc_fn(dict(energy_set), pathway)
            except KeyError as missing:
                failed.append({"row_id": cand["row_id"], "model": m,
                               "reason": f"missing species {missing.args[0]}"})
                continue
            dg_steps = [float(x) for x in dg_steps]
            results[m] = {
                "eta": float(eta),
                "dG_steps": dg_steps,
                "dG_cumulative": [float(x) for x in dg_cum],
                "pds": int(np.argmax(dg_steps)) if dg_steps else None,
            }
        if primary not in results or len(results) < 2:
            continue
        methods = [primary] + sorted(m for m in results if m != primary)
        committee = methods[1:]
        st = _stats([results[m]["eta"] for m in methods])
        pds = {m: results[m]["pds"] for m in methods}

        shift, dominant = {}, {}
        for m in committee:
            relax = {s: v / (2.0 * force_constant)
                     for s, v in (cand.get("force_sq_sum", {}).get(m) or {}).items()}
            coeffs = step_coefficients(calc_fn, cand["energies"][m], pathway)
            n_steps = len(results[m]["dG_steps"])
            best_step, best_val, best_species = None, -1.0, None
            for i in range(n_steps):
                contrib = {s: abs(coeffs[s][i]) * relax.get(s, 0.0) for s in coeffs}
                total = sum(contrib.values())
                if total > best_val:
                    best_step, best_val = i, total
                    best_species = max(contrib, key=contrib.get) if contrib else None
            shift[m] = best_val if best_step is not None else 0.0
            dominant[m] = {"step": best_step, "species": best_species,
                           "source": _species_source(best_species) if best_species else None,
                           "relax_energy": relax.get(best_species) if best_species else None}
        worst = max(shift, key=shift.get) if shift else None
        shift_max = shift[worst] if worst else 0.0
        limit = max(st["std"], eta_floor)
        flagged = shift_max > limit
        pds_agree = len(set(pds.values())) == 1

        if flagged:
            label = "geometry_disagreement"
        elif st["std"] <= eta_tolerance and pds_agree:
            label = "robust"
        else:
            label = "uncertain"

        per_candidate[cand["row_id"]] = {
            "bulk_uuid": cand["bulk_uuid"],
            "surface_id": cand["surface_id"],
            "eta": {m: results[m]["eta"] for m in methods},
            "dG_steps": {m: results[m]["dG_steps"] for m in methods},
            "dG_cumulative": {m: results[m]["dG_cumulative"] for m in methods},
            "pds": pds,
            "pds_agree": pds_agree,
            "eta_mean": st["mean"], "eta_std": st["std"],
            "eta_min": st["min"], "eta_max": st["max"],
            "dG_steps_std": [float(np.std([results[m]["dG_steps"][i] for m in methods], ddof=1))
                             for i in range(len(results[primary]["dG_steps"]))],
            "bias_primary": float(results[primary]["eta"]
                                  - np.mean([results[m]["eta"] for m in committee])),
            "n_models": len(methods),
            "max_force_ads": {m: (cand.get("max_force_ads", {}).get(m) or {}) for m in committee},
            "geometry_shift": shift,
            "geometry_shift_max": shift_max,
            "geometry_shift_model": worst,
            "geometry_limit": limit,
            "geometry_source": dominant[worst]["source"] if worst else None,
            "geometry_via": ({"species": dominant[worst]["species"], "model": worst,
                              "step": dominant[worst]["step"]} if worst else None),
            "geometry_disagreement": flagged,
            "label": label,
        }

    return {"per_candidate": per_candidate,
            "per_bulk": adsorption_bulk_summary(per_candidate, primary),
            "failed": failed}


def adsorption_bulk_summary(per_candidate, primary):
    """Per bulk: best candidate (lowest eta) of every model, whether they agree,
    and Kendall tau of each model's candidate ranking against the primary's.
    ``per_candidate`` as returned by ``committee_adsorption`` (also usable on the
    per-row results read back from the database)."""
    per_bulk = {}
    by_bulk = {}
    for row_id, res in per_candidate.items():
        by_bulk.setdefault(res["bulk_uuid"], {})[row_id] = res
    for bulk_uuid, cands in by_bulk.items():
        models = set.intersection(*(set(r["eta"]) for r in cands.values()))
        methods = [primary] + sorted(m for m in models if m != primary)
        rankings = {m: _ranking({rid: r["eta"][m] for rid, r in cands.items()}) for m in methods}
        best = {m: rankings[m][0] for m in methods}
        per_bulk[bulk_uuid] = {
            "methods": methods,
            "n_candidates": len(cands),
            "best": best,
            "best_eta": {m: cands[best[m]]["eta"][m] for m in methods},
            "best_agree": len(set(best.values())) == 1,
            "kendall_tau": {m: _kendall_tau(rankings[primary], rankings[m])
                            for m in methods if m != primary},
        }
    return per_bulk

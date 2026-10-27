"""Post-pipeline reporting for
``PhaseDiagramMLWorkChain -> SurfaceBuilderWorkChain -> AdsorbatesWorkChain``.

Each stage writes its results to a different table (``DBComposition.stable_struct``
for bulk hull stability, ``DBSurface`` for the relaxed slabs SurfaceBuilderWorkChain
kept, ``DBSurfaceMLAdsorbate`` for the reaction-path free energies AdsorbatesWorkChain
computed on top of those slabs). This module joins the three by ``structure_uuid``
so that, per candidate bulk, one place answers "is this bulk any good for this
reaction/reaction_path?": is it on/near the hull, which surfaces were stable enough
to keep, and what does the resulting free-energy diagram look like.

Public API:

    bulk_candidates(chemical_formula) -> [bulk dict, ...]  # now includes "source"/"mp_id"/"ml_bulk_model"
    surfaces_for_bulk(chemical_formula) -> {structure_uuid: [surface dict, ...]}  # now includes "n_atoms"/"area"/"shift"
    reaction_results(chemical_formula, reaction, reaction_path) -> {structure_uuid: [candidate dict, ...]}  # now includes "repeat"/"coverage" (adsorbate concentration, in ML)
    electronic_for_bulk(chemical_formula) -> {structure_uuid: band_info dict}  # no-DFT light screen (empty if OpticalScreenWorkChain not run); negative ML gaps clamped to 0
    synthesizability_for_bulk(chemical_formula) -> {structure_uuid: [prediction dict, ...]}  # CSLLM screen (empty if SynthesizabilityScreenWorkChain not run)
    phonon_for_bulk(chemical_formula) -> {structure_uuid: [phonon dict, ...]}  # MLIP phonons of predicted bulks (empty if PhononScreenWorkChain not run)
    step_labels(reaction, reaction_path) -> [str, ...] | None
    summarize(chemical_formula, reaction, reaction_path) -> [bulk summary dict, ...]  # each surface now also carries its OWN reaction_candidates/best_candidate
    report(chemical_formula, reaction, reaction_path, plot_dir=None) -> [bulk summary dict, ...]

Plotting (matplotlib imported lazily -- only needed if you call these):

    plot_free_energy_diagram(dg_cumulative, labels=None, ..., ax=None)
    plot_bulk_comparison(summaries, ax_ehull=None, ax_eta=None)
    plot_surface_fed(surface, reaction, reaction_path, ax=None)  # FED for one surface's best candidate
    plot_bulk_detail(summary, reaction, reaction_path)  # one FED panel per surface, stacked in a column
    save_report_figures(summaries, reaction, reaction_path, output_dir) -> {structure_uuid: path, ...}

HTML report (also lazily needs matplotlib; see result-sample.html for the reference layout):

    raw_data(chemical_formula, reaction, reaction_path, summaries=None) -> [bulk dict, ...]
        -> summarize()'s output, plus the full bulk structure (pymatgen
        Structure dict) and full surface slab structure (pymatgen Slab dict)
        for every bulk/surface -- everything render_html_report() writes to
        "raw_data.json" for the page's download button.

    render_html_report(chemical_formula, reaction, reaction_path, summaries=None, output_path="report.html")
        -> writes a self-contained HTML file (Tailwind/lucide chrome, plots
        embedded as base64 PNGs): a stable-bulks table (uuid + source), one
        surfaces table per bulk (miller index, surface energy, size, best
        candidate's supercell repeat), and one reaction-path diagram per
        surface (titled with its site + eta + coverage). The size (# atoms/
        area) shown per surface is scaled to its BEST candidate's own
        supercell repeat (see _repeat_multiplier()), not the bare relaxed
        slab -- different candidates on the same surface can use different
        repeats for different coverages, so this scaling is per-candidate,
        not a fixed per-surface number. Does NOT include the cross-bulk
        comparison chart (E-above-hull + best eta per bulk) -- that plot
        (plot_bulk_comparison()) is not rendered as part of the standard
        pipeline report; call it directly if you want it. Also writes
        "raw_data.json" (see raw_data()) next to output_path and
        links a download button to it. Also returns the HTML string.

Caveats worth knowing:

* Importing this module, and calling ``bulk_candidates``/``surfaces_for_bulk``/
  ``reaction_results``/``summarize``/``report(..., plot_dir=None)``, never
  needs an AiiDA profile -- they talk to the Postgres tables directly, like
  ``akmc_analysis.py``. Only ``step_labels()`` -- and therefore
  ``plot_bulk_detail()``/``save_report_figures()``/``report(..., plot_dir=...)``,
  which call it for FED x-axis labels -- lazily imports the matching
  ``uvsib.workchains.<reaction>`` module, and every one of those transitively
  imports ``uvsib.workflows.settings``, which DOES load an AiiDA profile. Run
  those specific calls from an environment where a profile is already
  available (e.g. the same process/script that ran the pipeline). This is why
  ``MainWorkChain.pipeline_report`` (``uvsib/workchains/main.py``) -- which
  calls ``report()`` and ``render_html_report()`` as its own outline step
  right after AKMC, writing into ``settings.REPORTS_DIR/<composition>_
  <reaction>_<reaction_path>/`` (set in ``run_dir/run.py``, not the
  per-run ``settings.run_dir``) -- works safely: it already
  runs inside a profile-loaded AiiDA worker process. ``render_html_report()`` ALSO always
  tries this same profile-loading import via ``_ml_surface_model()``/
  ``_ml_stage_head()`` (for the "ML Bulk/Surface Model" + task metadata
  fields), but defensively -- unlike ``step_labels()``, it catches the
  failure and just shows "&mdash;" rather than raising, so calling
  ``render_html_report()`` outside a profile-loaded environment still works
  (just without those fields). ``raw_data()`` itself stays DB-only like the
  functions above it -- no profile needed.
* "Surfaces found" means "slabs SurfaceBuilderWorkChain kept after relaxation
  and ranked by formation energy" -- it already dropped non-converged slabs
  before storing (see ``inspect_relax`` in ``surface_builder.py``); there is no
  further stability filter applied here.
* Every ``reaction_results`` row already passed AdsorbatesWorkChain's own
  screening (eta <= 2.0 eV, see ``reaction_map`` in ``adsorbates.py``). A bulk
  with zero reaction_results rows for a given reaction/reaction_path either had
  no candidate under that threshold, or is missing a pathway intermediate that
  dissociated during relaxation (see the ``KeyError`` handling in
  ``AdsorbatesWorkChain.store_results_ml``) -- not necessarily "bad", possibly
  "not evaluated". Which of these it was is recorded per surface by
  AdsorbatesWorkChain in ``DBSurface.attributes["adsorbates"][reaction][reaction_path]``
  and shown in the "Adsorbate screen" column (``adsorbate_diagnostic_parts``);
  a pathway with no candidate at all gets a summary of the reasons
  (``no_candidate_reasons``) and the report is still written.
"""
from collections import defaultdict

from uvsib.db.tables import DBComposition, DBSurface, DBSurfaceMLAdsorbate, DBSynthesizability, DBPhonon
from uvsib.db.utils import query_by_columns, query_structure
from uvsib.workchains.surface_uncertainty import adsorption_bulk_summary


def _bulk_source(structure_uuid, method=None):
    """Provenance of one bulk structure: the ``source``/``method``/``mp_id`` of
    its ``DBStructureVersion`` row (e.g. source="MPDB_stb"/"csp"/"generated",
    method="MACE"/..., mp_id="mp-1234" for MPDB sources -- see ``add_structures``
    in ``db/utils.py``). Prefers the version matching ``method`` (the
    composition's ``ml_bulk_model``, i.e. the version PhaseDiagramMLWorkChain
    actually ranked) when given and present; otherwise falls back to the first
    version found. Returns ``None`` if the structure has no stored version at
    all."""
    versions = query_structure({"uuid": structure_uuid})
    if not versions:
        return None
    # mp_id is a property of the underlying structure, not a given version --
    # the DFT (MPDB) version carries it, but the ML-relaxed version added later
    # (add_version_to_existing_structure in mpdb_ml.py) does not -- so resolve it
    # from whichever version has it rather than only the method-matched one.
    mp_id = next((v.mp_id for v in versions if v.mp_id), None)
    if method is not None:
        for v in versions:
            if v.method == method:
                return {"source": v.source, "method": v.method, "mp_id": mp_id}
    v = versions[0]
    return {"source": v.source, "method": v.method, "mp_id": mp_id}


def bulk_candidates(chemical_formula):
    """Every bulk structure PhaseDiagramMLWorkChain kept as stable for
    ``chemical_formula``, sorted by ascending E-above-hull (most stable
    first)."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return []
    stable_struct = rows[0].stable_struct or {}
    threshold = stable_struct.get("ml_ehull_threshold")
    model = stable_struct.get("ml_bulk_model")

    # committee E_hull uncertainty (EhullUncertaintyWorkChain); ignored if it
    # was computed for a different primary model than the current selection
    uncertainty = stable_struct.get("ml_uncertainty") or {}
    if uncertainty.get("primary") != model:
        uncertainty = {}
    uncertainty_meta = {k: v for k, v in uncertainty.items() if k != "per_uuid"} or None
    uncertainty_by_uuid = uncertainty.get("per_uuid") or {}

    candidates = []
    for entry in stable_struct.get("ml_selection", []):
        provenance = _bulk_source(entry["uuid"], method=model)
        unc = uncertainty_by_uuid.get(entry["uuid"])
        candidates.append({
            "uncertainty": unc,
            "uncertainty_meta": uncertainty_meta if unc else None,
            "structure_uuid": entry["uuid"],
            "ehull": entry["ehull"],
            "selected_above_threshold": entry.get("selected_above_threshold", False),
            "ehull_threshold": threshold,
            "ml_bulk_model": model,
            "source": provenance["source"] if provenance else None,
            "mp_id": provenance["mp_id"] if provenance else None,
        })
    candidates.sort(key=lambda c: c["ehull"])
    return candidates


def hull_gaps(chemical_formula):
    """Subsystems of ``chemical_formula`` whose structure generation is not
    Ready (Empty / Failed / Missing), as stored by PhaseDiagramMLWorkChain in
    ``DBComposition.stable_struct``: ``{chemsys: coverage}`` with the
    generation status and the number of MPDB / other competing structures
    the hull used there. Empty for runs predating the record."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return {}
    stable_struct = rows[0].stable_struct or {}
    coverage = stable_struct.get("hull_coverage") or {}
    return {cs: coverage.get(cs, {}) for cs in stable_struct.get("hull_gaps") or []}


_GAP_STATUS_TEXT = {
    "Empty": "the generators produced no structures",
    "Failed": "structure generation failed",
    "Missing": "no structure generation record",
}


def hull_gap_lines(gaps):
    """One sentence per hull gap, most serious first: no MPDB or other
    competing structure at all, then failed/missing generation."""
    def severity(item):
        cs, cov = item
        covered = (cov.get("n_mpdb") or 0) + (cov.get("n_other") or 0)
        return (covered > 0, cov.get("gen_status") == "Empty", cs)

    lines = []
    for cs, cov in sorted(gaps.items(), key=severity):
        status = cov.get("gen_status")
        text = _GAP_STATUS_TEXT.get(status, f"generation status {status}")
        n_mpdb, n_other = cov.get("n_mpdb") or 0, cov.get("n_other") or 0
        if n_mpdb or n_other:
            parts = [_n(n_mpdb, "Materials Project structure")] if n_mpdb else []
            parts += [_n(n_other, "other structure")] if n_other else []
            covered = "competing phases from " + " and ".join(parts)
        else:
            covered = "no competing phase in the hull"
        lines.append(f"{cs}: {text}; {covered}.")
    return lines


def _slab_area(slab):
    """Surface area (Å²) of a pymatgen ``Slab.as_dict()`` blob, from the
    in-plane lattice vectors a, b (|a x b|). Returns ``None`` if the slab has
    no lattice matrix."""
    matrix = (slab.get("lattice") or {}).get("matrix")
    if not matrix:
        return None
    a, b = matrix[0], matrix[1]
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return (cx ** 2 + cy ** 2 + cz ** 2) ** 0.5


def _analysed(attributes, key):
    """The committee ``uncertainty`` blob of a DBSurface / DBSurfaceMLAdsorbate
    row once the analysis has run on it (``key`` present), else ``None``."""
    unc = (attributes or {}).get("uncertainty") or {}
    return unc if key in unc else None


def surface_uncertainty_summary(chemical_formula):
    """``DBComposition.stable_struct["surface_uncertainty"]`` (run settings +
    per-bulk facet-ranking agreement), or ``None``."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    return ((rows[0].stable_struct or {}).get("surface_uncertainty") if rows else None) or None


def surfaces_for_bulk(chemical_formula):
    """DBSurface rows for ``chemical_formula``, grouped by bulk structure_uuid
    and ranked within each bulk by ascending surface formation energy (most
    stable surface first)."""
    rows = query_by_columns(DBSurface, {"composition": chemical_formula})
    by_uuid = defaultdict(list)
    for row in rows:
        slab = row.slab or {}
        by_uuid[str(row.structure_uuid)].append({
            "surface_id": row.id,
            "miller_index": slab.get("miller_index"),
            "formation_energy": row.formation_energy,
            "n_atoms": len(slab.get("sites") or []),
            "area": _slab_area(slab),
            "shift": slab.get("shift"),
            "uncertainty": _analysed(row.attributes, "gamma_std"),
            # {reaction: {reaction_path: diagnostics}} written by AdsorbatesWorkChain
            "adsorbate_runs": (row.attributes or {}).get("adsorbates"),
        })
    for surfaces in by_uuid.values():
        surfaces.sort(key=lambda s: s["formation_energy"]
                      if s["formation_energy"] is not None else float("inf"))
    return dict(by_uuid)


def _parse_repeat(repeat):
    """Parse a ``DBSurfaceMLAdsorbate.repeat`` value into an ``(nx, ny, nz)``
    int tuple (z is always 1; only nx, ny multiply in-plane coverage -- see
    ``get_multipliers()``/``make_supercell()`` in ``codes/files/adsorbates.py``).
    In practice (confirmed against the live DB), ``query_by_columns`` returns
    this ``Text`` column as a POSTGRES ARRAY LITERAL, e.g. ``"{1,2,1}"`` --
    curly braces, not JSON/Python list brackets. That string is deliberately
    NOT run through ``ast.literal_eval``: Python parses ``{1,2,1}`` as a SET
    literal, which silently collapses duplicate entries (``"{1,1,1}"`` ->
    ``{1}``, a 1-element set; ``"{2,2,1}"`` -> the wrong 2-element ``{1, 2}``)
    -- exactly the repeat values that matter most (no repeat, or equal x/y
    repeat) would then parse to the wrong shape or silently fail. Also
    tolerates an already-parsed list/tuple, or (for robustness against a
    differently-configured column) JSON/Python list syntax like
    ``"[1, 2, 1]"``. Returns ``None`` if unparseable."""
    if repeat is None:
        return None
    if isinstance(repeat, (list, tuple)):
        values = repeat
    else:
        text = str(repeat).strip()
        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1].strip()
            values = [v.strip() for v in inner.split(",")] if inner else []
        else:
            import ast
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
    try:
        return tuple(int(v) for v in values)
    except (TypeError, ValueError):
        return None


def _coverage(repeat):
    """Adsorbate coverage (surface concentration), in monolayers, from the
    in-plane (nx, ny) supercell repeat: one adsorbate per nx*ny repeated
    surface cells -> 1/(nx*ny) ML. Returns ``None`` if ``repeat`` is
    unparseable or has a zero in-plane multiplier."""
    parsed = _parse_repeat(repeat)
    if not parsed or len(parsed) < 2:
        return None
    nx, ny = parsed[0], parsed[1]
    if not nx or not ny:
        return None
    return 1.0 / (nx * ny)


def _coverage_label(repeat):
    """Human-readable coverage label, e.g. ``"1/4 ML (2x2)"``, from a raw
    ``repeat`` value. Returns ``None`` if ``repeat`` is unparseable."""
    parsed = _parse_repeat(repeat)
    if not parsed or len(parsed) < 2:
        return None
    nx, ny = parsed[0], parsed[1]
    if not nx or not ny:
        return None
    n = nx * ny
    return f"1/{n} ML ({nx}x{ny})" if n != 1 else "1 ML (1x1)"


def reaction_results(chemical_formula, reaction, reaction_path):
    """DBSurfaceMLAdsorbate rows for one (composition, reaction, reaction_path),
    grouped by bulk structure_uuid and ranked within each bulk by ascending eta
    (kinetically/thermodynamically best candidate first)."""
    rows = query_by_columns(DBSurfaceMLAdsorbate, {
        "composition": chemical_formula,
        "reaction": reaction,
        "reaction_path": reaction_path,
    })
    by_uuid = defaultdict(list)
    for row in rows:
        by_uuid[str(row.structure_uuid)].append({
            "row_id": row.id,
            "surface_id": row.surface_id,
            "miller_index": row.surface_miller_index,
            "site_type": row.site_type,
            "ads_coord": row.ads_coord,
            "eta": row.eta,
            "dG_steps": row.dG_steps,
            "dG_cumulative": row.dG_cumulative,
            "repeat": _parse_repeat(row.repeat),
            "coverage": _coverage(row.repeat),
            "uncertainty": _analysed(row.attributes, "eta_std"),
        })
    for candidates in by_uuid.values():
        candidates.sort(key=lambda c: c["eta"])
    return dict(by_uuid)


def electronic_for_bulk(chemical_formula):
    """``{structure_uuid: band_info}`` -- the no-DFT light-harvesting screen
    (ML gap + Butler--Ginley band edges + per-(reaction, pathway) straddle
    verdict) OpticalScreenWorkChain wrote onto ``DBStructureVersion.band_info``.

    Only the version PhaseDiagramMLWorkChain ranked (``method == ml_bulk_model``)
    is considered, matching ``bulk_candidates()``. Empty when the optical screen
    was not run. DB-only, no AiiDA profile needed."""
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    model = (rows[0].stable_struct or {}).get("ml_bulk_model") if rows else None

    out = {}
    for version in query_structure({"composition": chemical_formula}):
        if version.band_info is None:
            continue
        if model is not None and version.method != model:
            continue
        out[str(version.structure_uuid)] = _clamp_negative_gap(version.band_info)
    return out


def _clamp_negative_gap(band_info):
    """A negative ML gap is a model artefact, not physics: report it as 0 eV.
    Butler--Ginley edges are cb = chi - E_e - gap/2 and vb = cb + gap, so with
    gap = 0 both collapse onto their midpoint; the stored straddle margins are
    recomputed from those edges. The model's own value is kept as
    ``gap_raw_eV``. A negative per-model gap (``gap_values_eV``) is clamped to
    0 too, even when the mean is positive. Returns ``band_info`` unchanged when
    no gap is negative."""
    gap = (band_info or {}).get("gap_eV")
    per_model = (band_info or {}).get("gap_values_eV") or {}
    if (gap is None or gap >= 0) and all(v >= 0 for v in per_model.values()):
        return band_info
    from uvsib.workchains.redox_couples import straddle_verdict

    info = dict(band_info)
    info["gap_values_eV"] = {k: max(v, 0.0) for k, v in per_model.items()}
    if gap is None or gap >= 0:
        info["notes"] = list(info.get("notes") or []) + ["negative per-model ML gap clamped to 0"]
        return info
    info["gap_raw_eV"] = gap
    info["gap_eV"] = 0.0
    info["notes"] = list(info.get("notes") or []) + [
        f"negative ML gap ({gap:.4f} eV) clamped to 0; band edges set to mid-gap"]
    for key in ("band_edges_vs_rhe_V", "band_edges_vs_vacuum_eV"):
        edges = info.get(key)
        if edges and edges.get("cb") is not None and edges.get("vb") is not None:
            mid = round(0.5 * (edges["cb"] + edges["vb"]), 4)
            info[key] = {**edges, "cb": mid, "vb": mid}
    rhe = info.get("band_edges_vs_rhe_V")
    if rhe and info.get("straddle"):
        info["straddle"] = {
            reaction: {
                path: {**v, **straddle_verdict(rhe["cb"], rhe["vb"], v["u_red"], v["u_ox"],
                                               v.get("margin_required_V", 0.2))}
                for path, v in by_path.items()}
            for reaction, by_path in info["straddle"].items()}
    return info


def synthesizability_for_bulk(chemical_formula):
    """``{structure_uuid: [prediction dict, ...]}`` -- the DBSynthesizability
    rows SynthesizabilityScreenWorkChain wrote (one per model), sorted by
    model name. Empty when the screen was not run. DB-only, no AiiDA profile
    needed."""
    out = defaultdict(list)
    for row in query_by_columns(DBSynthesizability, {"composition": chemical_formula}):
        out[str(row.structure_uuid)].append({
            "model": row.synthesizability_model,
            "score": row.synthesizability_score,
            "label": row.synthesizability_label,
            "uncertainty": row.synthesizability_uncertainty,
            "method": row.predicted_synthesis_method,
            "precursors": row.predicted_precursors,
            "in_domain": row.in_domain,
            "ehull": row.ehull,
            "attributes": row.attributes or {},
        })
    for predictions in out.values():
        predictions.sort(key=lambda p: p["model"])
    return dict(out)


def phonon_for_bulk(chemical_formula):
    """``{structure_uuid: [phonon dict, ...]}`` -- the DBPhonon rows
    PhononScreenWorkChain wrote (one per model; predicted bulks only), sorted
    by model name. Empty when the screen was not run. DB-only, no AiiDA
    profile needed."""
    out = defaultdict(list)
    for row in query_by_columns(DBPhonon, {"composition": chemical_formula}):
        out[str(row.structure_uuid)].append({
            "model": row.phonon_model,
            "source": row.source,
            "status": row.status,
            "dynamically_stable": row.dynamically_stable,
            "min_frequency_THz": row.min_frequency_THz,
            "n_imaginary_qpoints": row.n_imaginary_qpoints,
            "e0": row.e0,
            "free_energy_300K": row.free_energy_300K,
            "ehull": row.ehull,
            "attributes": row.attributes or {},
        })
    for results in out.values():
        results.sort(key=lambda r: r["model"])
    return dict(out)


_OER_LABELS = ["*", "*OH", "*O", "*OOH", "O2 + *"]

_REACTION_MODULES = {
    "CO2RR": ("co2rr", "CO2RR_PATHWAYS"),
    "CER": ("cer", "CER_PATHWAYS"),
    "HER": ("her", "HER_PATHWAYS"),
    "NRR": ("nrr", "NRR_PATHWAYS"),
    "NOXRR": ("noxrr", "NOXRR_PATHWAYS"),
    "ORR": ("orr", "ORR_PATHWAYS"),
}


def adsorbate_diagnostic_parts(diag):
    """Short human-readable outcome of the adsorbate screen on one surface, from
    the diagnostics AdsorbatesWorkChain stores in
    ``DBSurface.attributes["adsorbates"][reaction][reaction_path]``. Shared by
    the workchain log and the HTML report so both say the same thing."""
    if not diag:
        return ["no diagnostics recorded"]
    if diag.get("job") == "not_screened":
        return [f"not screened (only the {diag.get('max_num_ads', '?')} lowest-energy "
                "surfaces of each bulk are screened)"]
    if diag.get("job") == "failed":
        return [f"adsorbate job failed (exit {diag.get('exit_status')})"]
    parts = []
    summary = diag.get("summary") or {}
    if summary:
        parts.append(f"{summary.get('n_sites', 0)} sites tried")
        for key, text in (("n_placement_failed", "not placeable (overlap)"),
                          ("n_relax_error", "relaxation error"),
                          ("n_not_converged", "not converged")):
            if summary.get(key):
                parts.append(f"{summary[key]} {text}")
    if summary.get("identity_check_skipped"):
        parts.append("identity check skipped for " + ", ".join(summary["identity_check_skipped"])
                     + " (bad reference geometry)")
    rejected = diag.get("rejected") or []
    n_rejected = sum(r["count"] for r in rejected)
    if n_rejected:
        detail = ", ".join(f"{r['adsorbate']} {r['reason']} ×{r['count']}" for r in rejected)
        parts.append(f"{n_rejected} rejected by sanity checks ({detail})")
    elif summary and not summary.get("validate", True):
        parts.append("sanity checks off")
    if diag.get("n_missing_intermediate"):
        names = ", ".join(sorted(diag.get("missing_intermediates") or {}))
        parts.append(f"{diag['n_missing_intermediate']} missing intermediate ({names})")
    if diag.get("n_eta_above"):
        limit = diag.get("eta_limit")
        text = (f"{diag['n_eta_above']} with η above {limit:g} V" if limit is not None
                else f"{diag['n_eta_above']} above the η limit")
        if diag.get("min_eta") is not None and not diag.get("n_stored"):
            text += f" (lowest {diag['min_eta']:.2f} V)"
        parts.append(text)
    parts.append(f"{diag.get('n_stored', 0)} stored")
    return parts


def no_candidate_reasons(summaries):
    """Why a (reaction, pathway) ended with no candidate at all: the per-surface
    diagnostics of every bulk added up, as a list of sentences."""
    surfaces = [surf for s in summaries for surf in s["surfaces"]]
    diags = [surf.get("adsorbate_diagnostics") for surf in surfaces]
    screened = [d for d in diags if d and d.get("job") in ("ok", "failed")]
    if not surfaces:
        return ["no surface was retained, so no adsorbate was placed"]
    if not screened:
        return ["no adsorbate diagnostics were recorded for this reaction path"]
    max_ads = next((d["max_num_ads"] for d in diags if d and d.get("max_num_ads")), None)
    reasons = [f"{len(screened)} of {_n(len(surfaces), 'surface')} screened"
               + (f" (only the {max_ads} lowest-energy surfaces of each bulk are screened)"
                  if max_ads else "")]
    failed = [d for d in screened if d.get("job") == "failed"]
    if failed:
        reasons.append(f"{_n(len(failed), 'adsorbate job')} failed")
    totals, rejected, missing = defaultdict(int), defaultdict(int), defaultdict(int)
    min_eta, limit = None, None
    for d in screened:
        for key, value in (d.get("summary") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
        for r in d.get("rejected") or []:
            rejected[(r["adsorbate"], r["reason"])] += r["count"]
        for name, n in (d.get("missing_intermediates") or {}).items():
            missing[name] += n
        totals["n_eta_above"] += d.get("n_eta_above") or 0
        if d.get("min_eta") is not None and (min_eta is None or d["min_eta"] < min_eta):
            min_eta = d["min_eta"]
        limit = d.get("eta_limit", limit)
    if totals["n_sites"]:
        reasons.append(f"{_n(totals['n_sites'], 'adsorption site')} tried, "
                       f"{_n(totals['n_relaxed'], 'complete set')} relaxed")
    for key, word, text in (("n_placement_failed", "site", "where an adsorbate could not be placed without overlap"),
                            ("n_relax_error", "set", "with a relaxation error"),
                            ("n_not_converged", "set", "that did not converge")):
        if totals[key]:
            reasons.append(f"{_n(totals[key], word)} {text}")
    if rejected:
        detail = ", ".join(f"{ads} {reason} ×{n}" for (ads, reason), n
                           in sorted(rejected.items(), key=lambda kv: -kv[1]))
        reasons.append(f"{_n(sum(rejected.values()), 'set')} rejected by the sanity checks ({detail})")
    if missing:
        reasons.append(f"{_n(sum(missing.values()), 'set')} missing a pathway intermediate "
                       f"({', '.join(sorted(missing))})")
    if totals["n_eta_above"]:
        bit = f" (lowest η {min_eta:.2f} V)" if min_eta is not None else ""
        lim = f"{limit:g} V" if limit is not None else "the limit"
        reasons.append(f"{_n(totals['n_eta_above'], 'candidate')} with η above {lim}{bit}")
    return reasons


def _n(count, word):
    """'1 set' / '3 sets'."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def _adsorbate_screen_cell(surf, esc):
    """Report-table cell: the adsorbate-screen outcome of one surface."""
    diag = surf.get("adsorbate_diagnostics")
    parts = adsorbate_diagnostic_parts(diag)
    if not diag and surf.get("reaction_candidates"):
        # screened by a run that predates the diagnostics
        css, title = "text-slate-600", "Screened before per-surface diagnostics were recorded"
        parts = [f"{len(surf['reaction_candidates'])} stored", "(screened before diagnostics were recorded)"]
    elif not diag:
        css, title = "text-slate-400 italic", ("No adsorbate diagnostics: the surface was not screened, or it "
                                               "was screened before diagnostics were recorded")
    elif diag.get("job") == "not_screened":
        css, title = "text-slate-400 italic", "Only the lowest-energy surfaces of each bulk are screened"
    elif diag.get("job") == "failed":
        css, title = "text-red-700", "The adsorbate calculation for this surface did not finish"
    elif not diag.get("n_stored"):
        css, title = "text-amber-700", "Screened, but no candidate was stored for this surface"
    else:
        css, title = "text-slate-600", "Outcome of the adsorbate screen on this surface"
    return (f'<div class="text-xs leading-5 {css}" title="{esc(title)}">'
            + "<br>".join(esc(p) for p in parts) + "</div>")


def step_labels(reaction, reaction_path):
    """Free-energy-diagram x-axis labels lining up with one candidate's
    ``dG_cumulative``, derived from the SAME ``*_PATHWAYS`` step dict
    AdsorbatesWorkChain used to compute it -- each step's newly formed
    ``*``-prefixed species (coefficient +1) -- so labels can never drift out of
    sync with the numbers. Returns ``None`` if the reaction/reaction_path is
    unrecognized. OER has no ``*_PATHWAYS`` dict (its steps are hard-coded in
    ``oer.py``), so its 5 labels are hard-coded here to match."""
    reaction = reaction.strip().upper()
    if reaction == "OER":
        return list(_OER_LABELS)

    if reaction not in _REACTION_MODULES:
        return None
    mod_name, dict_name = _REACTION_MODULES[reaction]
    import importlib
    module = importlib.import_module(f"uvsib.workchains.{mod_name}")
    pathways = getattr(module, dict_name)

    reaction_path = reaction_path.strip().lower()
    if reaction_path not in pathways:
        return None

    labels = ["*"]
    for step in pathways[reaction_path]["steps"][1:]:
        formed = [species for species, coeff in step.items()
                  if coeff == 1 and species.startswith("*")]
        labels.append(formed[0] if formed else f"step {len(labels)}")
    return labels


def summarize(chemical_formula, reaction, reaction_path):
    """Per-bulk report: hull stability, the surfaces SurfaceBuilderWorkChain
    found, and the reaction-path candidates AdsorbatesWorkChain stored on top
    of them. One dict per bulk structure_uuid, bulks with a known hull
    position sorted by ascending ehull first (any leftover uuid that only
    shows up in DBSurface/DBSurfaceMLAdsorbate -- e.g. a bulk selection that
    changed between pipeline reruns -- is appended after, with ``bulk=None``).

    Each surface dict under ``"surfaces"`` also carries its OWN
    ``"reaction_candidates"``/``"best_candidate"`` (the reaction_results rows
    for that specific surface_id, ranked by ascending eta) alongside the
    bulk-wide versions -- so a per-surface reaction-path plot always has
    exactly the candidates that surface produced, not the bulk's best overall."""
    bulks = {b["structure_uuid"]: b for b in bulk_candidates(chemical_formula)}
    surfaces_by_uuid = surfaces_for_bulk(chemical_formula)
    results_by_uuid = reaction_results(chemical_formula, reaction, reaction_path)
    electronic_by_uuid = electronic_for_bulk(chemical_formula)
    synthesizability_by_uuid = synthesizability_for_bulk(chemical_formula)
    phonon_by_uuid = phonon_for_bulk(chemical_formula)
    surface_unc = surface_uncertainty_summary(chemical_formula) or {}

    all_uuids = set(bulks) | set(surfaces_by_uuid) | set(results_by_uuid)
    summaries = []
    for uid in all_uuids:
        surfaces = surfaces_by_uuid.get(uid, [])
        candidates = results_by_uuid.get(uid, [])

        candidates_by_surface = defaultdict(list)
        for c in candidates:
            candidates_by_surface[c["surface_id"]].append(c)
        for surf in surfaces:
            # already eta-sorted: candidates arrives eta-sorted from
            # reaction_results(), and grouping preserves that order.
            surf_candidates = candidates_by_surface.get(surf["surface_id"], [])
            surf["reaction_candidates"] = surf_candidates
            surf["best_candidate"] = surf_candidates[0] if surf_candidates else None
            runs = surf.pop("adsorbate_runs", None) or {}
            surf["adsorbate_diagnostics"] = (runs.get(reaction) or {}).get(reaction_path)

        # per-bulk adsorption agreement, derived from the per-row results (the
        # adsorption analysis stores nothing composition-wide)
        analysed = {str(c["row_id"]): c["uncertainty"] for c in candidates if c.get("uncertainty")}
        adsorption_meta = next(iter(analysed.values()), {}).get("meta") if analysed else None
        adsorption_bulk = (adsorption_bulk_summary(analysed, adsorption_meta["primary"]).get(uid)
                           if adsorption_meta else None)

        summaries.append({
            "structure_uuid": uid,
            "bulk": bulks.get(uid),
            "surfaces": surfaces,
            "n_surfaces": len(surfaces),
            "best_surface_formation_energy": surfaces[0]["formation_energy"] if surfaces else None,
            "reaction_candidates": candidates,
            "n_reaction_candidates": len(candidates),
            "best_candidate": candidates[0] if candidates else None,
            "electronic": electronic_by_uuid.get(uid),
            "synthesizability": synthesizability_by_uuid.get(uid, []),
            "phonon": phonon_by_uuid.get(uid, []),
            "surface_uncertainty": (surface_unc.get("per_bulk") or {}).get(uid),
            "surface_uncertainty_meta": ({k: v for k, v in surface_unc.items() if k != "per_bulk"}
                                         if surface_unc else None),
            "adsorption_uncertainty": adsorption_bulk,
            "adsorption_uncertainty_meta": adsorption_meta,
        })

    summaries.sort(key=lambda s: (s["bulk"] is None,
                                   s["bulk"]["ehull"] if s["bulk"] else 0.0))
    return summaries


def report(chemical_formula, reaction, reaction_path, plot_dir=None):
    """Convenience entry point: ``summarize()``, optionally also rendering and
    saving the comparison + per-bulk figures to ``plot_dir`` (created if
    missing) and attaching their paths onto each summary as ``"figures"``."""
    summaries = summarize(chemical_formula, reaction, reaction_path)
    if plot_dir is not None:
        paths = save_report_figures(summaries, reaction, reaction_path, plot_dir)
        for summary in summaries:
            summary["figures"] = paths.get(summary["structure_uuid"])
    return summaries


################################################################################
# Plotting -- matplotlib is only imported once one of these is actually called.
################################################################################

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib (pip install matplotlib); "
            "the rest of this module works without it."
        ) from exc
    return plt


def plot_free_energy_diagram(dg_cumulative, labels=None, equilibrium_potential=None,
                              title=None, ax=None, committee=None):
    """Staircase free-energy diagram for one candidate's ``dG_cumulative``
    (the same values AdsorbatesWorkChain derived eta from). The
    potential-determining step -- the single largest rise, which sets eta -- is
    highlighted in red. ``committee`` ({model: dG_cumulative}, from the
    adsorption uncertainty analysis) overlays each committee MLIP's diagram as
    thin lines."""
    plt = _require_matplotlib()

    dg_cumulative = list(dg_cumulative)
    n = len(dg_cumulative)
    if labels is None:
        labels = [str(i) for i in range(n)]
    step_deltas = [dg_cumulative[i + 1] - dg_cumulative[i] for i in range(n - 1)]
    pds_index = max(range(len(step_deltas)), key=lambda i: step_deltas[i]) if step_deltas else None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(1.6 * n + 1, 4))

    for i, g in enumerate(dg_cumulative):
        on_pds = pds_index is not None and i in (pds_index, pds_index + 1)
        ax.hlines(g, i - 0.3, i + 0.3, linewidth=4, color="crimson" if on_pds else "steelblue")
    for i in range(n - 1):
        ax.plot([i + 0.3, i + 1 - 0.3], [dg_cumulative[i], dg_cumulative[i + 1]],
                linestyle="--", linewidth=2,
                color="crimson" if i == pds_index else "gray")

    if committee:
        styles = [("darkorange", ":"), ("purple", "-."), ("teal", (0, (5, 2))), ("olive", ":")]
        for k, (model, dg_m) in enumerate(sorted(committee.items())):
            color, ls = styles[k % len(styles)]
            dg_m = list(dg_m)
            for i, g in enumerate(dg_m):
                ax.hlines(g, i - 0.3, i + 0.3, linewidth=1.5, color=color, linestyle=ls,
                          label=model if i == 0 else None)
            for i in range(len(dg_m) - 1):
                ax.plot([i + 0.3, i + 1 - 0.3], [dg_m[i], dg_m[i + 1]], linewidth=0.8,
                        color=color, linestyle=ls)
        ax.legend(fontsize=8, loc="best")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(r"$\Delta G$ (eV)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)

    subtitle_parts = []
    if pds_index is not None:
        subtitle_parts.append(f"PDS: {labels[pds_index]} -> {labels[pds_index + 1]} "
                               f"({step_deltas[pds_index]:.2f} eV)")
    if equilibrium_potential is not None:
        subtitle_parts.append(f"U_eq = {equilibrium_potential:.2f} V")
    subtitle = "  |  ".join(subtitle_parts)
    # subtitle on its own line rather than appended to the same line -- a
    # single-line title (miller + site + eta + coverage + PDS + U_eq) easily
    # overflows a fixed-width panel and gets clipped at the right edge.
    full_title = f"{title or ''}\n{subtitle}" if subtitle else (title or "")
    ax.set_title(full_title, fontsize=11)

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_bulk_comparison(summaries, ax_ehull=None, ax_eta=None):
    """Two-panel across-bulk comparison for one composition: E-above-hull per
    bulk (with the ML selection threshold drawn in) and the best reaction-path
    eta found on each bulk -- the two numbers that most directly say which bulk
    is worth pursuing further."""
    plt = _require_matplotlib()

    own_fig = ax_ehull is None and ax_eta is None
    if own_fig:
        fig, (ax_ehull, ax_eta) = plt.subplots(1, 2, figsize=(max(6, 1.4 * len(summaries)), 4))

    tick_labels = [s["structure_uuid"][:8] for s in summaries]

    ehulls = [s["bulk"]["ehull"] if s["bulk"] else float("nan") for s in summaries]
    threshold = next((s["bulk"]["ehull_threshold"] for s in summaries if s["bulk"]), None)
    bar_colors = ["goldenrod" if (s["bulk"] and s["bulk"]["selected_above_threshold"]) else "seagreen"
                  for s in summaries]
    bars = ax_ehull.bar(tick_labels, ehulls, color=bar_colors)

    # committee E_hull uncertainty: one marker per committee MLIP + min-max
    # whisker; bars flagged for geometry disagreement are hatched
    meta = next((s["bulk"]["uncertainty_meta"] for s in summaries
                 if s["bulk"] and s["bulk"].get("uncertainty")), None)
    if meta:
        markers = ["o", "s", "^", "D", "v", "P"]
        for k, model in enumerate(m["model"] for m in meta.get("committee", [])):
            xs, ys = [], []
            for i, s in enumerate(summaries):
                unc = s["bulk"].get("uncertainty") if s["bulk"] else None
                if unc and model in unc["ehull"]:
                    xs.append(i)
                    ys.append(unc["ehull"][model])
            ax_ehull.scatter(xs, ys, marker=markers[k % len(markers)], color="black",
                             s=22, zorder=3, label=model)
        for i, s in enumerate(summaries):
            unc = s["bulk"].get("uncertainty") if s["bulk"] else None
            if not unc:
                continue
            ax_ehull.vlines(i, unc["ehull_min"], unc["ehull_max"], color="black",
                            linewidth=1, zorder=2)
            if unc.get("geometry_disagreement"):
                bars[i].set_hatch("//")
                bars[i].set_edgecolor("purple")

    if threshold is not None:
        ax_ehull.axhline(threshold, color="black", linestyle="--", linewidth=1,
                          label=f"threshold = {threshold:.2f}")
    if threshold is not None or meta:
        ax_ehull.legend(fontsize=8)
    ax_ehull.set_ylabel("E above hull (eV/atom)")
    ax_ehull.set_title("Bulk stability")
    ax_ehull.tick_params(axis="x", rotation=45)

    best_eta = [s["best_candidate"]["eta"] if s["best_candidate"] else float("nan") for s in summaries]
    ax_eta.bar(tick_labels, best_eta, color="steelblue")
    ax_eta.set_ylabel(r"best $\eta$ (V)")
    ax_eta.set_title("Best reaction-path candidate per bulk")
    ax_eta.tick_params(axis="x", rotation=45)

    if own_fig:
        fig.tight_layout()
        return fig, (ax_ehull, ax_eta)
    return ax_ehull, ax_eta


# Human-readable names for (reaction, reaction_path), used in the report
# title. Mirrors the pathway table on the platform's Reaction Pathways
# documentation page; unknown pairs fall back to "REACTION / path".
_PATHWAY_TITLES = {
    "HER": {
        "volmer_tafel": "Hydrogen evolution (Volmer–Tafel)",
        "volmer_heyrovsky": "Hydrogen evolution (Volmer–Heyrovsky)",
    },
    "OER": {
        "4e_mechanism": "Oxygen evolution (4e⁻ mechanism)",
        "default": "Oxygen evolution (4e⁻ mechanism)",
        "none": "Oxygen evolution (4e⁻ mechanism)",
    },
    "ORR": {
        "4e_associative": "O₂ reduction to H₂O (4e⁻ associative)",
        "4e_dissociative": "O₂ reduction to H₂O (4e⁻ dissociative)",
        "2e_to_h2o2": "O₂ reduction to H₂O₂ (2e⁻)",
    },
    "CER": {
        "volmer_tafel": "Chlorine evolution (Volmer–Tafel)",
        "volmer_heyrovsky": "Chlorine evolution (Volmer–Heyrovsky)",
        "krishtalik": "Chlorine evolution (Krishtalik)",
    },
    "CO2RR": {
        "co2_to_co": "CO₂ reduction to CO",
        "co2_to_hcooh": "CO₂ reduction to HCOOH",
        "co_to_ch4": "CO reduction to CH₄",
        "co_to_ch3oh": "CO reduction to CH₃OH",
        "co2_to_ch4": "CO₂ reduction to CH₄",
        "co2_to_ch3oh": "CO₂ reduction to CH₃OH",
        "co2_to_c2h4": "CO₂ reduction to C₂H₄",
    },
    "NOXRR": {
        "no_dissociative": "NO dissociation to N₂",
        "no_to_nh3_noh": "NO reduction to NH₃ (via *NOH)",
        "no_to_nh3_nhoh": "NO reduction to NH₃ (via *NHOH)",
        "no_to_n2o": "NO reduction to N₂O",
        "no2_to_no": "NO₂ reduction to NO",
        "no3_to_nh3": "NO₃⁻ reduction to NH₃",
        "no3_to_n2": "NO₃⁻ reduction to N₂",
    },
    "NRR": {
        "distal": "N₂ reduction to NH₃ (distal)",
        "alternating": "N₂ reduction to NH₃ (alternating)",
        "dissociative": "N₂ reduction to NH₃ (dissociative)",
    },
}


def pathway_title(reaction, reaction_path):
    """Readable name for (reaction, reaction_path), e.g. ("CO2RR",
    "co2_to_hcooh") -> "CO₂ reduction to HCOOH"; "REACTION / path" if the
    pair is not in ``_PATHWAY_TITLES``."""
    rxn = (reaction or "").strip().upper()
    rp = (reaction_path or "").strip().lower()
    return _PATHWAY_TITLES.get(rxn, {}).get(rp) or f"{reaction} / {reaction_path}"


# What each optical-screen gap model (``codes/files/electronic.py``) predicts:
# key -> (model, training data, target gap). ``megnet_mfi``'s target is the
# fidelity chosen at run time (``megnet_fidelity_label``), so it is None here.
_GAP_MODEL_INFO = {
    "megnet_mfi": ("MEGNet multi-fidelity", None, None),
    "alignn_pbe": ("ALIGNN", "Materials Project", "PBE"),
    "alignn_mbj": ("ALIGNN", "JARVIS-DFT", "TBmBJ"),
}


def _gap_model_targets(electronic):
    """``[(key, model, training data, target gap), ...]`` for the gap models
    that ran on this bulk (``electronic`` = its ``band_info`` blob)."""
    out = []
    for key in electronic.get("gap_models") or []:
        model, data, target = _GAP_MODEL_INFO.get(key, (key, None, None))
        if key == "megnet_mfi":
            target = electronic.get("megnet_fidelity_label")
        out.append((key, model, data, target))
    return out


def _reaction_couple(reaction, reaction_path):
    """The redox couple (``{"role","u_red","u_ox","label"}``) the band gap must
    straddle for this (reaction, reaction_path), or ``None``. Lazily imports
    ``uvsib.workchains.redox_couples`` (AiiDA-profile-loading, like
    ``step_labels``); returns ``None`` if that import fails."""
    try:
        from uvsib.workchains.redox_couples import couple_for
        return couple_for(reaction, reaction_path)
    except Exception:
        return None


def bulk_straddle(electronic, reaction, reaction_path):
    """The stored straddle verdict for this (reaction, reaction_path) out of a
    bulk's ``electronic`` (``band_info``) blob, tolerant of the OER
    ``default``/``none`` spellings. ``None`` if the optical screen was not run
    or produced no band edges."""
    if not electronic:
        return None
    by_path = (electronic.get("straddle") or {}).get((reaction or "").strip().upper(), {})
    if not by_path:
        return None
    rp = (reaction_path or "").strip().lower()
    if rp in by_path:
        return by_path[rp]
    if len(by_path) == 1:
        return next(iter(by_path.values()))
    return by_path.get("default")


def bulk_straddle_by_model(electronic, reaction, reaction_path):
    """``{model_key: straddle_verdict}`` for this (reaction, reaction_path), with
    each gap model's own gap in place of the mean. Derived from the mean-gap
    verdict's couple/margin and ``gap_values_eV`` rather than read from the
    stored ``straddle_by_model``, so bulks screened before that field existed
    get it too. Empty when there is no mean-gap verdict."""
    verdict = bulk_straddle(electronic, reaction, reaction_path)
    edges = (electronic or {}).get("band_edges_vs_rhe_V")
    if not verdict or not edges:
        return {}
    from uvsib.workchains.redox_couples import model_band_edges, straddle_verdict

    return {key: straddle_verdict(e["cb"], e["vb"], verdict["u_red"], verdict["u_ox"],
                                  verdict["margin_required_V"])
            for key, e in model_band_edges(edges, electronic.get("gap_values_eV")).items()}


def straddle_consensus(by_model):
    """``("straddles" | "model-dependent" | "no", [passing model keys])`` over the
    per-model verdicts of ``bulk_straddle_by_model``; ``(None, [])`` when empty."""
    if not by_model:
        return None, []
    passing = [key for key, v in by_model.items() if v.get("straddles")]
    if len(passing) == len(by_model):
        return "straddles", passing
    return ("model-dependent" if passing else "no"), passing


def plot_band_alignment(summaries, reaction, reaction_path, ax=None):
    """One vertical CB->VB bar per bulk on an inverted potential axis (V vs RHE,
    increasing downward, as in the usual band diagram), with the reaction's
    reduction/oxidation levels drawn as dashed lines. Bars span the mean-gap
    edges; they are green when every gap model's gap straddles the couple with
    margin, orange when only some do, grey otherwise. Returns ``None`` if no
    bulk has band edges (optical screen not run)."""
    plt = _require_matplotlib()

    edged = [s for s in summaries
             if (s.get("electronic") or {}).get("band_edges_vs_rhe_V")]
    if not edged:
        return None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(edged)), 4.5))

    couple = _reaction_couple(reaction, reaction_path)
    labels = []
    for i, s in enumerate(edged):
        e = s["electronic"]
        cb = e["band_edges_vs_rhe_V"]["cb"]
        vb = e["band_edges_vs_rhe_V"]["vb"]
        consensus, _ = straddle_consensus(bulk_straddle_by_model(e, reaction, reaction_path))
        colour = {"straddles": "seagreen", "model-dependent": "orange"}.get(consensus, "silver")
        ax.bar(i, vb - cb, bottom=cb, width=0.55, color=colour, edgecolor="black", linewidth=0.6)
        gap = e.get("gap_eV")
        ax.text(i, cb, f" {gap:.2f} eV" if gap is not None else "", ha="center", va="bottom", fontsize=8)
        labels.append(s["structure_uuid"][:8])

    if couple:
        ax.axhline(couple["u_red"], color="crimson", linestyle="--", linewidth=1,
                   label=f'reduction {couple["u_red"]:.2f} V')
        ax.axhline(couple["u_ox"], color="royalblue", linestyle="--", linewidth=1,
                   label=f'oxidation {couple["u_ox"]:.2f} V')
        ax.legend(fontsize=8, loc="best")

    ax.set_xticks(range(len(edged)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Potential (V vs RHE)")
    ax.invert_yaxis()
    ax.set_title(f"ML band edges vs {reaction}/{reaction_path} redox window", fontsize=10)

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_surface_fed(surface, reaction, reaction_path, ax=None):
    """Free-energy diagram for ONE surface's best (lowest-eta)
    reaction-path candidate -- the per-surface counterpart to
    ``plot_free_energy_diagram``, which plots one already-chosen candidate's
    numbers. Every stable surface gets its own diagram, since different
    surfaces of the same bulk can favor different sites/PDS."""
    plt = _require_matplotlib()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4, 4))

    miller = str(tuple(surface["miller_index"])) if surface["miller_index"] else "?"
    best = surface.get("best_candidate")
    if best is None:
        ax.text(0.5, 0.5, "no reaction-path\ncandidates", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(miller, fontsize=10)
    else:
        labels = step_labels(reaction, reaction_path) or \
            [str(i) for i in range(len(best["dG_cumulative"]))]
        coverage_bit = _coverage_label(best.get("repeat"))
        coverage_bit = f", {coverage_bit}" if coverage_bit else ""
        unc = best.get("uncertainty")
        committee = None
        eta_bit = f"{best['eta']:.2f} V"
        if unc:
            primary = (unc.get("meta") or {}).get("primary")
            committee = {m: v for m, v in unc["dG_cumulative"].items() if m != primary}
            eta_bit = f"{best['eta']:.2f} ± {unc['eta_std']:.2f} V"
        plot_free_energy_diagram(
            best["dG_cumulative"], labels=labels,
            title=f"{miller}  site={best['site_type']}, eta={eta_bit}{coverage_bit}",
            ax=ax, committee=committee,
        )

    if own_fig:
        fig.tight_layout()
        return fig, ax
    return ax


def plot_bulk_detail(summary, reaction, reaction_path):
    """Per-bulk figure: one reaction-path free-energy diagram (via
    ``plot_surface_fed``) for each surface that actually HAS a
    reaction-path candidate, stacked in a SINGLE COLUMN. A row layout would
    rescale every panel to fit a fixed width, so a bulk with 5 surfaces gets
    cramped panels while a bulk with 1 gets an oversized one -- the number of
    surfaces per bulk is unpredictable, so a column keeps each panel the
    same fixed, readable size regardless of count; only the figure's total
    height grows. Surfaces with no candidate are skipped rather than padded
    with an empty "no candidates" panel. Surface formation energy is already
    in the surfaces table alongside this figure, so it is not duplicated
    here as a bar chart."""
    plt = _require_matplotlib()

    fed_surfaces = [s for s in summary["surfaces"] if s["best_candidate"]]
    n_fed_panels = len(fed_surfaces)
    ehull_bit = f"E_hull = {summary['bulk']['ehull']:.3f} eV/atom" if summary["bulk"] else "E_hull unknown"

    # tight_layout() does not reserve room for suptitle() -- without an
    # explicit rect, the bulk-level suptitle collides with (renders on top
    # of) the topmost panel's own (2-line) title. Reserve a FIXED number of
    # inches at the top regardless of total figure height, since that height
    # scales with n_fed_panels and a fixed top *fraction* would over- or
    # under-reserve depending on panel count.
    suptitle_inches = 0.55

    if not n_fed_panels:
        fig_height = 3.0
        fig, ax = plt.subplots(figsize=(6, fig_height))
        ax.text(0.5, 0.5, "no reaction-path candidates for this bulk's surfaces",
                ha="center", va="center", wrap=True)
        ax.set_axis_off()
        fig.suptitle(f"{summary['structure_uuid'][:8]}  ({ehull_bit})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 1 - suptitle_inches / fig_height))
        return fig

    fig_height = 4.8 * n_fed_panels
    fig, axes = plt.subplots(n_fed_panels, 1, figsize=(6.5, fig_height))
    if n_fed_panels == 1:
        axes = [axes]
    for ax, surface in zip(axes, fed_surfaces):
        plot_surface_fed(surface, reaction, reaction_path, ax=ax)

    fig.suptitle(f"{summary['structure_uuid'][:8]}  ({ehull_bit})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - suptitle_inches / fig_height))
    return fig


def save_report_figures(summaries, reaction, reaction_path, output_dir):
    """Render and save one detail figure per bulk (one reaction-path FED
    panel per surface, stacked in a column) into ``output_dir`` (created if
    missing). Returns ``{structure_uuid: path, ...}``."""
    import os
    plt = _require_matplotlib()
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    for summary in summaries:
        fig = plot_bulk_detail(summary, reaction, reaction_path)
        detail_path = os.path.join(output_dir, f"bulk_{summary['structure_uuid'][:8]}.png")
        fig.savefig(detail_path, dpi=150)
        plt.close(fig)
        paths[summary["structure_uuid"]] = detail_path

    return paths


################################################################################
# HTML report -- mirrors the layout of result-sample.html (Tailwind + lucide).
# Plots are embedded as base64 PNGs so the output is a single, shareable file.
################################################################################

def _fig_to_data_uri(fig):
    """Render a matplotlib figure to a base64 PNG data URI and close it."""
    import io
    import base64
    plt = _require_matplotlib()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _ml_surface_model():
    """The face-build (surface relaxation) ML model name, read the SAME way
    SurfaceBuilderWorkChain does -- ``settings.inputs["face_build"]["model"]``
    (see ``_model_cmdline()`` in ``workchains/surface_builder.py``). This is a
    run-level config value, not a per-row DB field, so it lazily imports
    ``uvsib.workflows.settings`` -- which DOES load an AiiDA profile (see the
    module-level caveat above) -- and returns ``None`` if that fails (e.g.
    called outside a profile-loaded environment)."""
    try:
        from uvsib.workflows import settings
        return settings.inputs["face_build"]["model"]
    except Exception:
        return None


def _ml_stage_head(stage):
    """The ``head`` (task) an ML stage's model was run with -- e.g. UMA's
    "omat" bulk-relax head vs "oc20" face-build head -- read the SAME way
    ``_model_cmdline()`` (``workchains/surface_builder.py``) passes it on as
    ``--task_name`` for face-build (``bulk_relax`` is analogous; see
    ``input.yaml``'s ``bulk_relax``/``face_build`` blocks, each a
    ``model``/``head``/... dict). ``stage`` is ``"bulk_relax"`` or
    ``"face_build"``. Like ``_ml_surface_model()``, this is a run-level
    config value, not a per-row DB field, so it lazily imports
    ``uvsib.workflows.settings`` (AiiDA-profile-loading) and returns ``None``
    if that fails, or if the stage has no ``head`` configured."""
    try:
        from uvsib.workflows import settings
        return settings.inputs[stage].get("head")
    except Exception:
        return None


def _model_label(model, task):
    """Display label combining an ML model name with its task/head, e.g.
    ``"UMA (omat)"`` -- or just the model name if no task is available.
    Returns ``None`` if ``model`` itself is unavailable."""
    if model is None:
        return None
    return f"{model} ({task})" if task else model


def _repeat_multiplier(parsed_repeat):
    """In-plane atom/area multiplier (``nx * ny``) implied by an already
    ``_parse_repeat``-d ``(nx, ny, nz)`` repeat tuple -- the factor a
    surface's base (1x1, as stored on ``DBSurface``) ``n_atoms``/``area``
    must be scaled by to match the actual supercell a given reaction
    candidate was computed on (``make_supercell((nx, ny, 1))`` in
    ``generate_adsorbed_structures()``, ``codes/files/adsorbates.py`` --
    z never repeats). Different candidates on the SAME surface can use
    different repeats (different coverages), so this must be applied
    per-candidate, not once per surface. Returns 1 (no scaling, i.e. the
    base cell) if ``parsed_repeat`` is ``None``/unparseable/has a zero
    in-plane multiplier."""
    if not parsed_repeat or len(parsed_repeat) < 2:
        return 1
    nx, ny = parsed_repeat[0], parsed_repeat[1]
    return nx * ny if nx and ny else 1


def raw_data(chemical_formula, reaction, reaction_path, summaries=None):
    """Comprehensive raw-data export for one (composition, reaction,
    reaction_path): everything ``summarize()`` computes (ehull, surface
    formation energies, eta, dG, coverage, ...) PLUS the full bulk structure
    (pymatgen ``Structure.as_dict()``, from ``DBStructureVersion.structure``)
    and full surface slab structure (pymatgen ``Slab.as_dict()``, from
    ``DBSurface.slab``) for every bulk/surface listed in the report -- the
    underlying data behind every number the HTML page shows, for anyone who
    wants to reproduce or further analyze it outside this report.

    ``summaries`` lets a caller reuse an already-computed ``summarize()``
    result instead of re-querying the database."""
    if summaries is None:
        summaries = summarize(chemical_formula, reaction, reaction_path)

    bulk_structures = {}
    for s in summaries:
        uid = s["structure_uuid"]
        model = s["bulk"]["ml_bulk_model"] if s["bulk"] else None
        versions = query_structure({"uuid": uid})
        version = next((v for v in versions if v.method == model), versions[0]) if versions else None
        bulk_structures[uid] = version.structure if version else None

    slab_by_surface_id = {
        row.id: row.slab
        for row in query_by_columns(DBSurface, {"composition": chemical_formula})
    }

    data = []
    for s in summaries:
        entry = dict(s)
        entry["bulk_structure"] = bulk_structures.get(s["structure_uuid"])
        entry["surfaces"] = [
            {**surf, "slab_structure": slab_by_surface_id.get(surf["surface_id"])}
            for surf in s["surfaces"]
        ]
        data.append(entry)
    return data


_SYNTH_BADGE = {
    "synthesizable": "bg-green-100 text-green-700",
    "not synthesizable": "bg-red-100 text-red-700",
    "uncertain": "bg-amber-100 text-amber-700",
}

_METHOD_LABELS = {
    "solid_state": "solid-state",
    "solution": "solution",
    "solid_state&solution": "solid-state &amp; solution",
}


def _precursor_check(top, esc):
    """&#10003; / &#9888;&#xFE0E; marker for the top-ranked precursor set."""
    if top.get("element_consistent"):
        return ('<span class="ml-1 text-green-600" title="every target element is '
                'supplied; no non-volatile foreign elements">&#10003;</span>')
    why = []
    if top.get("missing_elements"):
        why.append("missing " + ", ".join(top["missing_elements"]))
    if top.get("extra_elements"):
        why.append("foreign " + ", ".join(top["extra_elements"]))
    if top.get("unparsable"):
        why.append("unparsable " + ", ".join(top["unparsable"]))
    return f'<span class="ml-1 text-red-600" title="{esc("; ".join(why))}">&#9888;&#xFE0E;</span>'


def _method_cell(p, esc):
    """Compact per-row method cell (only used when polymorphs disagree)."""
    method = p.get("method")
    if not method:
        return "&mdash;"
    method_probs = (p.get("attributes") or {}).get("method_probabilities") or {}
    prob = method_probs.get(method)
    prob_bit = f' <span class="text-slate-400">({prob:.2f})</span>' if prob is not None else ""
    title = ", ".join(f"{k}: {v:.2f}" for k, v in method_probs.items())
    return f'<span title="{esc(title)}">{_METHOD_LABELS.get(method, esc(method))}{prob_bit}</span>'


def _precursor_cell(p, esc):
    """Compact per-row precursor cell (only used when polymorphs disagree)."""
    sets = [x for x in (p.get("precursors") or []) if x.get("precursors")]
    if not sets:
        return "&mdash;"
    top = sets[0]
    alternatives = " | ".join(" + ".join(x["precursors"]) for x in sets[1:])
    alt_bit = (f'<div class="text-xs text-slate-400" title="lower-ranked precursor sets">'
               f'alt: {esc(alternatives)}</div>') if alternatives else ""
    return (f'<span class="font-mono text-xs">{esc(" + ".join(top["precursors"]))}</span>'
            f'{_precursor_check(top, esc)}{alt_bit}')


def _method_paragraph(formula, p, esc):
    """Prose for the composition-level method prediction."""
    method = p.get("method")
    if not method:
        return (f"<p><strong>Synthesis method.</strong> No method prediction is available "
                f"for {esc(formula)}.</p>")
    method_probs = (p.get("attributes") or {}).get("method_probabilities") or {}
    prob = method_probs.get(method)
    prob_bit = f" (P&nbsp;=&nbsp;{prob:.2f})" if prob is not None else ""
    others = [f"{_METHOD_LABELS.get(k, esc(k))} {v:.2f}"
              for k, v in sorted(method_probs.items(), key=lambda kv: -kv[1]) if k != method]
    others_bit = f"; the alternatives score {', '.join(others)}" if others else ""
    return (f"<p><strong>Synthesis method.</strong> The model predicts "
            f"<strong>{_METHOD_LABELS.get(method, esc(method))}</strong> synthesis for "
            f"{esc(formula)}{prob_bit}{others_bit}.</p>")


def _precursor_paragraph(formula, p, esc):
    """Prose for the composition-level precursor prediction: the parsed sets
    when there are any, otherwise the model's top free-text answer."""
    all_sets = p.get("precursors") or []
    sets = [x for x in all_sets if x.get("precursors")]
    domain_bit = ""
    if p.get("in_domain") is False:
        domain_bit = (' <span class="px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 text-[10px] font-bold" '
                      'title="more elements than the precursor model was validated on">'
                      'out of domain</span>')
    if sets:
        top = sets[0]
        alts = [esc(" + ".join(x["precursors"])) for x in sets[1:]]
        alt_bit = (f" Lower-ranked alternatives: "
                   + "; ".join(f'<span class="font-mono">{a}</span>' for a in alts) + ".") if alts else ""
        return (f"<p><strong>Precursors.</strong>{domain_bit} The top-ranked precursor set is "
                f'<span class="font-mono">{esc(" + ".join(top["precursors"]))}</span>'
                f"{_precursor_check(top, esc)}.{alt_bit}</p>")
    raw = next((x.get("raw") for x in all_sets if x.get("raw")), None)
    if not raw:
        return (f"<p><strong>Precursors.</strong>{domain_bit} No precursor prediction is "
                f"available for {esc(formula)}.</p>")
    raw = raw.strip()
    cut = "" if raw[-1:] in ".!?)" else "&hellip;"
    return (f"<p><strong>Precursors.</strong>{domain_bit} None of the {len(all_sets)} "
            f"beam-search outputs could be parsed into a precursor list; the model "
            f"answered with a free-text synthesis description instead. The top-ranked "
            f"output reads:</p>"
            f'<blockquote class="border-l-2 border-slate-300 pl-3 text-slate-600 italic '
            f'whitespace-pre-line">{esc(raw)}{cut}</blockquote>')


def _synthesizability_section(summaries, esc):
    """``(html, models_label)`` for the "Synthesizability" block -- one row per
    (bulk, model) from ``summary["synthesizability"]``. ``("", None)`` when
    SynthesizabilityScreenWorkChain wrote nothing for this composition, so
    the block is simply absent.

    Method and precursors are predicted from the composition, so when every
    polymorph of a model agrees they are written once as prose above the
    table instead of being repeated per row; the per-row columns come back
    only if they actually differ. The model column is shown only when more
    than one model ran (the model name is already in Pipeline Metadata)."""
    if not any(s.get("synthesizability") for s in summaries):
        return "", None

    import json

    predictions_all = [p for s in summaries for p in s.get("synthesizability") or []]
    models = sorted({p["model"] for p in predictions_all})
    show_model = len(models) > 1

    def comp_key(p):
        attrs = p.get("attributes") or {}
        return json.dumps([p.get("method"), attrs.get("method_probabilities"),
                           p.get("precursors"), p.get("in_domain")], sort_keys=True, default=str)

    shared = {m: len({comp_key(p) for p in predictions_all if p["model"] == m}) == 1 for m in models}
    per_row = not all(shared.values())

    flag_rows = {}
    for s in summaries:
        for p in s.get("synthesizability") or []:
            attrs = p.get("attributes") or {}
            flags = []
            if per_row and p.get("in_domain") is False:
                flags.append('<span class="px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 text-[10px] font-bold" '
                             'title="more elements than the precursor model was validated on">'
                             'precursors out of domain</span>')
            if attrs.get("selected_above_threshold"):
                flags.append('<span class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 text-[10px] font-bold" '
                             'title="kept only because the phase diagram must return at least one bulk">'
                             'above E<sub>hull</sub> threshold</span>')
            for note in attrs.get("notes") or []:
                flags.append(f'<span class="text-[10px] text-red-600">{esc(note)}</span>')
            flag_rows[id(p)] = flags
    show_notes = any(flag_rows.values())

    n_cols = 6 + show_model + 2 * per_row + show_notes
    rows = []
    for s in summaries:
        uid = s["structure_uuid"]
        b = s["bulk"]
        ehull_cell = f'{b["ehull"]:.4f}' if b else "&mdash;"
        source_cell = esc(_source_label(b.get("source"))) if b else "&mdash;"
        predictions = s.get("synthesizability") or []
        if not predictions:
            rows.append(
                f'<tr class="border-t"><td class="px-4 py-3 font-mono text-xs">{esc(uid[:8])}</td>'
                f'<td class="px-4 py-3">{ehull_cell}</td><td class="px-4 py-3">{source_cell}</td>'
                f'<td class="px-4 py-3 text-slate-400 italic" colspan="{n_cols - 3}">not screened</td></tr>')
            continue
        for p in predictions:
            score = p.get("score")
            score_cell = f"{score:.3f}" if score is not None else "&mdash;"
            unc = p.get("uncertainty")
            unc_cell = f"{unc:.2f}" if unc is not None else "&mdash;"
            label = p.get("label")
            if label:
                verdict_cell = (f'<span class="px-2 py-0.5 rounded {_SYNTH_BADGE.get(label, "bg-slate-100 text-slate-700")} '
                                f'text-xs font-bold whitespace-nowrap">{esc(label)}</span>')
            else:
                verdict_cell = "&mdash;"
            flags = flag_rows[id(p)]
            cells = [f'<td class="px-4 py-3 font-mono text-xs" title="{esc(uid)}">{esc(uid[:8])}</td>',
                     f'<td class="px-4 py-3">{ehull_cell}</td>',
                     f'<td class="px-4 py-3">{source_cell}</td>']
            if show_model:
                cells.append(f'<td class="px-4 py-3">{esc(p["model"])}</td>')
            cells += [f'<td class="px-4 py-3">{score_cell}</td>',
                      f'<td class="px-4 py-3">{verdict_cell}</td>',
                      f'<td class="px-4 py-3">{unc_cell}</td>']
            if per_row:
                cells += [f'<td class="px-4 py-3">{_method_cell(p, esc)}</td>',
                          f'<td class="px-4 py-3">{_precursor_cell(p, esc)}</td>']
            if show_notes:
                cells.append('<td class="px-4 py-3"><div class="flex flex-col gap-1">'
                             + "".join(flags) + "</div></td>")
            rows.append('\n                      <tr class="border-t align-top">'
                        + "".join(cells) + "</tr>")

    headers = ["Bulk", "E<sub>hull</sub> (eV/atom)", "Source"]
    if show_model:
        headers.append("Model")
    headers += ["P(synth.)", "Verdict", "Uncertainty (bits)"]
    if per_row:
        headers += ["Method", "Precursors"]
    if show_notes:
        headers.append("Notes")
    header_html = "".join(f'<th class="px-4 py-3">{h}</th>' for h in headers)

    thresholds = next((p["attributes"].get("thresholds") for p in predictions_all
                       if p.get("attributes")), None) or {}
    score_thr = thresholds.get("score")
    unc_thr = thresholds.get("uncertainty_bits")
    if score_thr is not None and unc_thr is not None:
        verdict_def = (f"<em>uncertain</em> if the uncertainty exceeds {unc_thr:g} bits; otherwise "
                       f"<em>synthesizable</em> if P &ge; {score_thr:g} and <em>not synthesizable</em> below.")
    else:
        verdict_def = ("<em>synthesizable</em> / <em>not synthesizable</em> / <em>uncertain</em>, from P(synth.) "
                       "and its uncertainty.")

    def term(dt, dd):
        return f"""
              <div>
                <dt class="font-semibold text-slate-900">{dt}</dt>
                <dd>{dd}</dd>
              </div>"""

    column_notes = [term("Bulk / E<sub>hull</sub> / Source",
                         """Bulk structure id, its ML-predicted energy above the convex hull (eV/atom), and
                  where it came from (MPDB(exp.): experimentally observed structure from the Materials Project;
                  MPDB(stb.): computationally stable structure from the Materials Project; CSP: crystal structure
                  prediction; generated: ML-generated). Hover the pointer over a bulk id
                  to see its full UUID.""")]
    if show_model:
        column_notes.append(term("Model", "Synthesizability model that made the prediction."))
    column_notes += [
        term("P(synth.)", "The model's probability (0&ndash;1) that the structure can be made experimentally."),
        term("Verdict", verdict_def),
        term("Uncertainty (bits)", "Binary entropy of P(synth.): 0 = confident, 1 bit = coin flip."),
    ]
    if per_row:
        column_notes += [
            term("Method", "Predicted synthesis method and its probability. Hover the pointer over a method to "
                           "see the probabilities of all methods."),
            term("Precursors", "Top-ranked precursor set, with lower-ranked sets below it. &#10003; / "
                               "&#9888;&#xFE0E;: the set is element-checked against the target; hover the pointer "
                               "over &#9888;&#xFE0E; to see what is missing or foreign."),
        ]
    if show_notes:
        column_notes.append(term("Notes", "<em>precursors out of domain</em>: more elements than the precursor "
                                          "model was validated on. <em>above E<sub>hull</sub> threshold</em>: kept "
                                          "only because the phase diagram must return at least one bulk."))
    column_notes_html = ('\n            <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm '
                         'text-slate-600 mt-5">' + "".join(column_notes) + "\n            </dl>")

    composition_html = ""
    if not per_row:
        formula = next(((p.get("attributes") or {}).get("formula") for p in predictions_all
                        if (p.get("attributes") or {}).get("formula")), "this composition")
        blocks = []
        for m in models:
            p = next(p for p in predictions_all if p["model"] == m)
            head = (f'<p class="text-xs font-bold text-slate-400 uppercase tracking-wider">{esc(m)}</p>'
                    if show_model else "")
            blocks.append(head + _method_paragraph(formula, p, esc) + _precursor_paragraph(formula, p, esc))
        composition_html = f"""
            <div class="text-sm text-slate-700 space-y-3 mb-5">
              <p class="text-slate-500">Method and precursors are predicted from the composition,
                so every polymorph of {esc(formula)} shares them.</p>
              {"".join(blocks)}
            </div>"""
        shared_note = ""
    else:
        shared_note = (" <strong>Method</strong> and <strong>precursors</strong> differ between "
                       "polymorphs here, so they are listed per row.")

    html = f"""
        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="test-tube" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Synthesizability</h2>
          </div>
          <div class="bg-white border rounded-xl p-6">
            <div class="border-l-4 border-primary bg-blue-50 text-slate-700 text-sm rounded-r-lg p-4 mb-5 space-y-2">
              <p>
                Predicted with {esc(", ".join(models))} for every bulk that passed the E<sub>above hull</sub> screen: whether
                the structure can be made experimentally, and by which method and from which precursors.{shared_note}
              </p>
              <p class="text-sm text-slate-500">
                Advisory only &mdash; no structure is removed on this basis. Precursor sets are
                element-checked against the target (&#10003; / &#9888;&#xFE0E;), not stoichiometry-balanced;
                CSLLM's precursor model was validated on binary/ternary compounds, so larger systems
                are marked out of domain.
              </p>
            </div>{composition_html}
            <div class="overflow-x-auto border rounded-lg">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 text-xs">
                  <tr>{header_html}</tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
              </table>
            </div>{column_notes_html}
          </div>
        </section>"""
    return html, ", ".join(models)


_STABILITY_BADGES = {
    "robust": ("bg-green-100 text-green-700", "robust",
               "E_hull below the threshold for every model"),
    "uncertain": ("bg-amber-100 text-amber-700", "uncertain",
                  "the models disagree on whether E_hull is below the threshold"),
    "unstable": ("bg-red-100 text-red-700", "unstable",
                 "E_hull above the threshold for every model"),
    "geometry_disagreement": ("bg-purple-100 text-purple-700", "relaxation",
                              "relaxing on a committee model could shift its E_hull by as much "
                              "as the model spread -- the spread is not trusted on its own"),
}

_RELAX_SOURCE_TEXT = {"self": "relaxation (self)", "competing": "relaxation (competing phase)"}


def _relax_via_text(unc):
    """``"via 3cc347b8 (GRACE)"`` / ``"self (GRACE)"`` for the structure that
    dominates the relaxation sensitivity; ``""`` if unknown."""
    via = unc.get("geometry_via")
    if not via:
        return ""
    if unc.get("geometry_source") == "self":
        return f'self ({via["model"]})'
    return f'via {via["uuid"][:8]} ({via["model"]})'


def _mev(value, digits=1):
    return _fmt(None if value is None else 1000 * value, digits)


def _relax_title(unc):
    """Tooltip: the relaxation sensitivity (force + stress) of every committee
    model, the limit and every contributing structure."""
    total, limit = unc.get("geometry_shift_max"), unc.get("geometry_limit")
    if total is None:
        return ""
    lines = [f"relaxation sensitivity {1000 * total:.1f} meV/atom "
             f"(limit {1000 * limit:.1f} = max(spread, floor))"
             f" under {unc.get('geometry_shift_model')}"]
    forces, stresses = unc.get("geometry_shift_force") or {}, unc.get("geometry_shift_stress") or {}
    for m, t in (unc.get("geometry_shift") or {}).items():
        lines.append(f"  {m}: force {_mev(forces.get(m))} + stress {_mev(stresses.get(m))} "
                     f"= {1000 * t:.1f} meV/atom")
    for c in unc.get("geometry_culprits") or []:
        lines.append(f'{"self" if c.get("self") else "competing"} {c["uuid"][:8]}: '
                     f'{1000 * c["shift"]:.1f} meV/atom = force {_mev(c.get("shift_force"))} '
                     f'+ stress {_mev(c.get("shift_stress"))} (x {c["fraction"]:.2f}); '
                     f'F={_fmt(c.get("max_force"), 2)} eV/A, '
                     f'stress={_fmt(c.get("max_stress"), 2)} GPa, P={_fmt(c.get("pressure"), 2)} GPa')
    return "\n".join(lines)


def _relax_cells(unc, esc):
    """``(dE_force, dE_stress, dE_total / limit)`` cells of one bulk row; on a
    flagged bulk the larger part is highlighted."""
    total = unc.get("geometry_shift_max")
    if total is None:
        return ['<td class="px-4 py-3">&mdash;</td>'] * 3
    flagged = unc.get("geometry_disagreement")
    title = esc(_relax_title(unc))
    force, stress = unc.get("geometry_shift_force_max"), unc.get("geometry_shift_stress_max")
    big = None
    if flagged and force is not None and stress is not None:
        big = "force" if force >= stress else "stress"
    cells = [f'<td class="px-4 py-3{" font-semibold text-purple-700" if big == name else ""}">'
             f'<span title="{title}">{_mev(value)}</span></td>'
             for name, value in (("force", force), ("stress", stress))]
    via = _relax_via_text(unc)
    via_bit = (f'<br><span class="text-xs font-normal text-slate-500">{esc(via)}</span>'
               if via and flagged else "")
    css = " bg-purple-50 text-purple-700 font-semibold" if flagged else ""
    cells.append(f'<td class="px-4 py-3{css}"><span title="{title}">{1000 * total:.1f} / '
                 f'{1000 * unc["geometry_limit"]:.1f}</span>{via_bit}</td>')
    return cells


def _uncertainty_methods(meta):
    """Primary first, then the committee (JSONB does not keep key order)."""
    return [meta["primary"]] + [m["model"] for m in meta.get("committee", [])]


def _join_names(names):
    """"A", "A and B", "A, B, and C"."""
    names = list(names)
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _stability_badge(unc, esc):
    """Colored label for one bulk's committee verdict (``&mdash;`` if none)."""
    if not unc:
        return "&mdash;"
    css, text, title = _STABILITY_BADGES.get(unc.get("label"), ("bg-slate-100 text-slate-600",
                                                               unc.get("label"), ""))
    if unc.get("label") == "geometry_disagreement":
        text = _RELAX_SOURCE_TEXT.get(unc.get("geometry_source"), text)
        detail = _relax_title(unc)
        if detail:
            title += "\n" + detail
    return (f'<span class="px-2 py-0.5 rounded {css} text-xs font-bold" '
            f'title="{esc(title)}">{esc(text)}</span>')


def _fmt(value, digits):
    return "&mdash;" if value is None else f"{value:.{digits}f}"


def _ehull_with_spread(bulk, esc):
    """E_hull table cell: primary value, &plusmn; std of the signed hull distance
    energy over all models when the committee analysis exists; per-model values
    in the tooltip."""
    unc, meta = bulk.get("uncertainty"), bulk.get("uncertainty_meta")
    if not unc or not meta:
        return f'{bulk["ehull"]:.4f}'
    per_model = " · ".join(f'{m} {unc["ehull"][m]:.3f}' for m in _uncertainty_methods(meta)
                           if m in unc["ehull"])
    title = (f"{per_model} (eV/atom). ± = std of the signed hull distance over "
             f"{unc['n_models']} MLIPs; range {unc['ehull_min']:.3f}–{unc['ehull_max']:.3f}")
    return (f'<span title="{esc(title)}">{bulk["ehull"]:.4f} &plusmn; '
            f'{unc["ed_std"]:.3f}</span>')


_PHONON_SOURCE_NAMES = {"csp": "CSP", "generated": "generated"}


def _phonon_section(summaries, esc):
    """``(html, models_label)`` for the "Lattice Dynamics" block -- one row
    per (bulk, model) from ``summary["phonon"]``, for the predicted (CSP /
    generated) bulks only; bulks without a phonon result are not listed.
    ``("", None)`` when there is nothing to show."""
    def shown(results):
        return [r for r in results or [] if r.get("source") in _PHONON_SOURCE_NAMES]

    results_all = [r for s in summaries for r in shown(s.get("phonon"))]
    if not results_all:
        return "", None
    def model_head(r):
        return _model_label(r["model"], (r.get("attributes") or {}).get("head"))

    models = sorted({model_head(r) for r in results_all})
    show_model = len(models) > 1

    params = next(((r.get("attributes") or {}).get("run_params") for r in results_all
                   if (r.get("attributes") or {}).get("run_params")), None) or {}
    tol = params.get("imaginary_tol_THz")
    tol_txt = f"{tol:g}" if tol is not None else "tol"
    mesh = params.get("mesh")
    mesh_txt = f"{mesh}&times;{mesh}&times;{mesh}" if mesh else "uniform"
    n_vol = len(params.get("volume_scales") or [1.0])
    qha = n_vol > 1
    # fixed-volume harmonic -> Helmholtz F; only the QHA minimisation over V
    # at P = 0 gives a Gibbs free energy
    fe_sym = "&Delta;G<sub>QHA</sub>" if qha else "F<sub>vib</sub>"

    rows = []
    n_flagged = 0
    for s in summaries:
        uid = s["structure_uuid"]
        b = s["bulk"]
        ehull_cell = f'{b["ehull"]:.4f}' if b else "&mdash;"
        for r in shown(s.get("phonon")):
            attrs = r.get("attributes") or {}
            if r.get("status") != "ok":
                verdict = ('<span class="px-2 py-0.5 rounded bg-slate-100 text-slate-600 text-xs font-bold">'
                           'failed</span>')
            elif r.get("dynamically_stable"):
                verdict = ('<span class="px-2 py-0.5 rounded bg-green-100 text-green-700 text-xs font-bold '
                           'whitespace-nowrap">dynamically stable</span>')
            else:
                verdict = ('<span class="px-2 py-0.5 rounded bg-red-100 text-red-700 text-xs font-bold '
                           'whitespace-nowrap">imaginary modes</span>')
            fmin = r.get("min_frequency_THz")
            fmin_cell = f"{fmin:.2f}" if fmin is not None else "&mdash;"
            n_imag, n_q = r.get("n_imaginary_qpoints"), attrs.get("n_qpoints")
            imag_cell = f"{n_imag} / {n_q}" if n_imag is not None and n_q else "&mdash;"
            f300 = r.get("free_energy_300K")
            f300_cell = f"{f300 * 1000:.1f}" if f300 is not None else "&mdash;"
            if f300 is not None and r.get("status") == "ok" and not r.get("dynamically_stable"):
                n_flagged += 1
                f300_cell = (f'<span class="text-slate-400" title="Imaginary modes were left out of the '
                             f'sum (contribute zero); not a meaningful free energy of this configuration">'
                             f'{f300_cell}&dagger;</span>')
            sc = attrs.get("supercell_matrix")
            sc_cell = ("&times;".join(str(n) for n in sc)
                       + (f' <span class="text-slate-400">({attrs.get("n_atoms_supercell")} at.)</span>'
                          if attrs.get("n_atoms_supercell") else "")) if sc else "&mdash;"
            if r.get("status") != "ok" and attrs.get("error"):
                sc_cell = f'<span class="text-[10px] text-red-600">{esc(attrs["error"])}</span>'
            cells = [f'<td class="px-4 py-3 font-mono text-xs" title="{esc(uid)}">{esc(uid[:8])}</td>',
                     f'<td class="px-4 py-3">{ehull_cell}</td>',
                     f'<td class="px-4 py-3">{esc(_PHONON_SOURCE_NAMES.get(r.get("source"), r.get("source") or ""))}</td>']
            if show_model:
                cells.append(f'<td class="px-4 py-3">{esc(model_head(r))}</td>')
            cells += [f'<td class="px-4 py-3">{verdict}</td>',
                      f'<td class="px-4 py-3">{fmin_cell}</td>',
                      f'<td class="px-4 py-3">{imag_cell}</td>',
                      f'<td class="px-4 py-3">{f300_cell}</td>',
                      f'<td class="px-4 py-3">{sc_cell}</td>']
            rows.append('\n                      <tr class="border-t align-top">'
                        + "".join(cells) + "</tr>")

    headers = ["Bulk", "E<sub>hull</sub> (eV/atom)", "Source"]
    if show_model:
        headers.append("Model")
    headers += ["Verdict", "Lowest &nu; (THz)", "Imag. q-points",
                f"{fe_sym}(300 K) (meV/atom)", "Supercell"]
    header_html = "".join(f'<th class="px-4 py-3">{h}</th>' for h in headers)

    tol_bit = (f" A mode below &minus;{tol_txt} THz away from &Gamma; counts as imaginary."
               if tol is not None else "")
    method_bit = (f"quasi-harmonic approximation over {n_vol} volumes" if qha
                  else "harmonic approximation at the fixed, relaxed volume")
    if qha:
        fe_def = ("is the quasi-harmonic Gibbs free-energy correction G(300 K) &minus; E<sub>0</sub>, "
                  "with G(T) = min<sub>V</sub> [E<sub>0</sub>(V) + F<sub>vib</sub>(V, T)] at P = 0 "
                  "(includes zero-point energy and the static-energy change from thermal expansion)")
    else:
        fe_def = ("is the harmonic Helmholtz vibrational free energy at fixed volume, "
                  "F<sub>vib</sub>(300 K) = F(300 K) &minus; E<sub>0</sub>, including the zero-point "
                  "energy. No volume relaxation is done, so this is a Helmholtz, not a Gibbs, free energy")
    min_len = params.get("min_supercell_length")
    min_len_bit = f", at least {min_len:g} &Aring; along each axis (at most 3&times;)" if min_len else ""
    model_def = ("""
                <div>
                  <dt class="font-semibold text-slate-900">Model</dt>
                  <dd>MLIP used for the phonon forces.</dd>
                </div>""" if show_model else "")
    fe_hover = (" Values marked &dagger; have imaginary modes left out of the sum; hover the pointer over "
                "such a value for details." if n_flagged else "")
    column_notes = f"""
              <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600 mt-5">
                <div>
                  <dt class="font-semibold text-slate-900">Bulk / E<sub>hull</sub> / Source</dt>
                  <dd>Bulk structure id, its ML-predicted energy above the convex hull (eV/atom), and
                    where it came from (CSP: crystal structure prediction; generated: ML-generated). Hover the pointer over a bulk id
                    to see its full UUID.</dd>
                </div>{model_def}
                <div>
                  <dt class="font-semibold text-slate-900">Verdict</dt>
                  <dd><em>dynamically stable</em>: no imaginary mode anywhere on the q-point mesh
                    (Lowest &nu; &ge; &minus;{tol_txt} THz). <em>imaginary modes</em>: at least one q-point has
                    an imaginary mode, so the structure is a saddle point of the MLIP energy surface and would
                    lower its energy by distorting along that mode. <em>failed</em>: the phonon calculation did
                    not finish (the error is shown under Supercell).</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Lowest &nu; (THz)</dt>
                  <dd>Lowest phonon frequency over the mesh, with &Gamma; left out (its three acoustic
                    modes are zero up to numerical noise). Imaginary frequencies are shown as negative
                    numbers, following phonopy: &minus;2.0 means 2.0i THz. The more negative the value, the
                    stronger the instability.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Imag. q-points</dt>
                  <dd><em>n</em> / <em>N</em>: of the <em>N</em> symmetry-irreducible q-points of the
                    {mesh_txt} sampling mesh (&Gamma; excluded), <em>n</em> have at least one branch below
                    &minus;{tol_txt} THz. It counts q-points, not modes, and each irreducible point counts once
                    whatever its weight, so it shows <em>how widespread</em> the instability is, not the fraction
                    of the Brillouin-zone volume. A few unstable points near &Gamma; often come from numerical
                    noise or a too-small supercell; many points, or zone-boundary points, point to a real
                    instability such as a lattice distortion or a cell doubling.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">{fe_sym}(300 K) (meV/atom)</dt>
                  <dd>{fe_sym}(300 K) {fe_def}. Per atom, at 300 K.{fe_hover}</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Supercell</dt>
                  <dd>Diagonal supercell used for the finite displacements{min_len_bit}, and its atom
                    count. Force constants beyond half the supercell are cut off, so a small supercell can
                    give spurious imaginary modes.</dd>
                </div>
              </dl>"""
    flag_note = (f"""
              <p class="text-sm text-slate-600 mt-4">
                <strong>&dagger; Structures with imaginary modes.</strong> The harmonic free energy is only
                defined for real frequencies. phonopy computes {fe_sym} by summing over modes with
                &nu; &gt; 0 and leaves the imaginary modes out (they contribute zero; the sum is not
                rescaled for the missing modes). The value shown is therefore not a free energy of that
                configuration and should not be compared with the dynamically stable structures. A
                meaningful value would need the structure to be relaxed along the unstable mode first, or an
                anharmonic treatment.
              </p>""" if n_flagged else "")

    html = f"""
        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="activity" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Lattice Dynamics</h2>
          </div>
          <div class="bg-white border rounded-xl p-6">
            <div class="border-l-4 border-primary bg-blue-50 text-slate-700 text-sm rounded-r-lg p-4 mb-5 space-y-2">
              <p>
                Phonons (finite displacements with the MLIP, {method_bit}) for the
                <strong>predicted</strong> bulks that passed the E<sub>above hull</sub> screen;
                Materials Project structures are not recomputed. Imaginary modes mean the
                structure is a saddle point, not a minimum, of the MLIP energy surface.{tol_bit}
                <strong>{fe_sym}(300 K)</strong> {fe_def}.
              </p>
              <p class="text-sm text-slate-500">
                No structure is removed on this basis.
              </p>
            </div>
            <div class="overflow-x-auto border rounded-lg">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 text-xs">
                  <tr>{header_html}</tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
              </table>
            </div>{flag_note}{column_notes}
          </div>
        </section>"""
    return html, ", ".join(models)


_SOURCE_LABELS = {"MPDB_exp": "MPDB(exp.)", "MPDB_stb": "MPDB(stb.)", "csp": "CSP"}


def _source_label(source):
    """Display name of a ``DBStructureVersion.source`` in the stable-bulks
    table; sources without an entry are shown as stored."""
    return _SOURCE_LABELS.get(source, source)


def _uncertainty_section(summaries, esc):
    """"Stability Sensitivity" section from the committee E_hull analysis
    (``bulk["uncertainty"]``). ``""`` when no bulk carries one."""
    with_unc = [s for s in summaries if s["bulk"] and s["bulk"].get("uncertainty")]
    if not with_unc:
        return ""
    meta = with_unc[0]["bulk"]["uncertainty_meta"]
    methods = _uncertainty_methods(meta)
    primary = meta["primary"]
    floor = meta.get("relax_energy_floor")

    head_cells = "".join(
        f'<th rowspan="2" class="px-4 py-3 align-bottom"><span class="normal-case">{esc(m)}{"*" if m == primary else ""}</span></th>'
        for m in methods)
    rows = []
    for s in with_unc:
        unc = s["bulk"]["uncertainty"]
        model_cells = []
        for m in methods:
            ehull = unc["ehull"].get(m)
            ed = unc["ed"].get(m)
            if ehull is None:
                model_cells.append('<td class="px-4 py-3">&mdash;</td>')
                continue
            ed_bit = f' <span class="text-slate-400">({ed:.3f})</span>' if ed is not None and ed < 0 else ""
            model_cells.append(f'<td class="px-4 py-3">{ehull:.3f}{ed_bit}</td>')
        rows.append(f"""
                  <tr class="border-t">
                    <td class="px-4 py-3 font-mono text-xs" title="{esc(s['structure_uuid'])}">{esc(s['structure_uuid'][:8])}</td>
                    {"".join(model_cells)}
                    <td class="px-4 py-3">{unc['ed_std']:.3f}</td>
                    <td class="px-4 py-3">{unc['bias_primary']:+.3f}</td>
                    <td class="px-4 py-3">{unc['stable_votes']}/{unc['n_models']}</td>
                    {"".join(_relax_cells(unc, esc))}
                    <td class="px-4 py-3">{_stability_badge(unc, esc)}</td>
                  </tr>""")

    committee_parts = [f'{esc(m["model"])}' + (f' ({esc(m["head"])})' if m.get("head") else "")
                       for m in meta.get("committee", [])]
    notes = []
    if meta.get("dropped"):
        notes.append(f'{len(meta["dropped"])} structure(s) missing for at least one model were '
                     "excluded from every model's hull.")
    conflicts = sum(len(v) for v in (meta.get("conflicts") or {}).values())
    if conflicts:
        notes.append(f"{conflicts} structure(s) already had a relaxed committee-model version and "
                     "were excluded.")
    if meta.get("missing_targets"):
        notes.append(f'{len(meta["missing_targets"])} selected bulk(s) could not be analysed.')
    notes_html = "".join(f"<p>{n}</p>" for n in notes)
    floor_txt = _fmt(1000 * floor if floor is not None else None, 1)
    column_notes = f"""
            <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600 mt-5">
              <div>
                <dt class="font-semibold text-slate-900">Bulk</dt>
                <dd>Bulk structure id (first 8 characters). Hover the pointer over an id to see the full UUID.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">{" / ".join(esc(m) + ("*" if m == primary else "") for m in methods)} (eV/atom)</dt>
                <dd>E<sub>hull</sub> of the bulk under each model; * marks the primary model used in the rest of
                  the workflow. For a bulk on the hull, the value in parentheses is the margin by which it beats its
                  best competitor (another polymorph or a decomposition), shown as a negative number.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">&sigma;(&Delta;E<sub>d</sub>) (eV/atom)</dt>
                <dd>&Delta;E<sub>d</sub> is the signed distance to the hull: E<sub>hull</sub> for a bulk above the
                  hull, the negative margin for a bulk on it. &sigma;(&Delta;E<sub>d</sub>) is its standard deviation
                  over all models, i.e. how strongly the models disagree about this bulk.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Bias (eV/atom)</dt>
                <dd>Primary minus committee mean of &Delta;E<sub>d</sub>. Positive: {esc(primary)} places the bulk higher
                  above the hull (less stable) than the committee does; negative: lower (more stable).</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Votes</dt>
                <dd>Number of models, out of all models, with E<sub>hull</sub> &le; {_fmt(meta.get('ehull_threshold'), 2)} eV/atom (the selection
                  threshold).</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">&Delta;E<sub>force</sub> / &Delta;E<sub>stress</sub> (meV/atom)</dt>
                <dd>The two parts of the relaxation sensitivity for model m*, the committee model with the largest
                  total (formula above). A large &Delta;E<sub>force</sub> points to residual forces on the atoms
                  (internal coordinates not relaxed); a large &Delta;E<sub>stress</sub> points to an imperfectly
                  relaxed cell shape. On a flagged bulk the larger part is highlighted. Hover the pointer over a
                  value to see the split for every committee model.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">&Delta;E<sub>total</sub> / limit (meV/atom)</dt>
                <dd>&Delta;E<sub>total</sub> = &Delta;E<sub>force</sub> + &Delta;E<sub>stress</sub>, compared with
                  limit = max(&sigma;(&Delta;E<sub>d</sub>), {floor_txt} meV/atom). Above the limit the bulk is flagged
                  (purple), and the structure that dominates is named under the value; it need not be one of the
                  bulks listed here. Hover the pointer over a value for every contributing structure.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Label</dt>
                <dd><em>robust</em>: E<sub>hull</sub> below the threshold for every model.
                  <em>uncertain</em>: the models disagree on whether E<sub>hull</sub> is below the threshold.
                  <em>unstable</em>: E<sub>hull</sub> above the threshold for every model.
                  <em>relaxation (self)</em> / <em>relaxation (competing phase)</em>: &Delta;E<sub>total</sub>
                  exceeds the limit, so relaxing on a committee model could shift E<sub>hull</sub> by as much as the
                  model spread, and the spread is not trusted on its own; <em>self</em> when the bulk's own structure
                  dominates, <em>competing phase</em> when a phase it is compared against does. Hover the pointer
                  over a <em>relaxation</em> label for the per-model details.</dd>
              </div>
            </dl>"""

    return f"""
        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="activity" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Stability Sensitivity</h2>
          </div>
          <div class="bg-white border rounded-xl p-6">
            <div class="border-l-4 border-primary bg-blue-50 text-slate-700 text-sm rounded-r-lg p-4 mb-5 space-y-2">
              <p>
                How much does the stability verdict depend on the choice of MLIP? E<sub>hull</sub> of each
                selected bulk is computed with the primary MLIP (<strong>{esc(primary)}</strong>) and a committee of
                {_join_names(committee_parts)}. Committee energies are <strong>single points on the
                {esc(primary)}-relaxed geometries</strong>; every model's hull is built from the same {meta.get('n_common')} structures.
                The spread between models measures their disagreement; it is not an error bar calibrated
                against DFT.
              </p>
              <p>
                <strong>Relaxation sensitivity</strong> (meV/atom): the committee single points are not
                taken at each model's own minimum. From the single-point forces and the
                <em>deviatoric</em> stress (a uniform pressure is a systematic lattice offset that largely
                cancels in E<sub>hull</sub>) the energy each committee model would gain by relaxing is
                estimated harmonically, for the bulk and for every phase it is compared against on that
                model's hull. It is an estimated energy correction, not a displacement of the atoms, and
                is split into two parts:
              </p>
              <div class="bg-white border rounded-lg px-4 py-3 font-mono text-xs leading-6 overflow-x-auto">
                &Delta;E<sub>force</sub>(s, m) = &lang;|F<sub>i</sub>|<sup>2</sup>&rang; / (2k)
                &nbsp;&nbsp;&nbsp; &Delta;E<sub>stress</sub>(s, m) = V<sub>atom</sub> &middot; |&sigma;<sub>dev</sub>|<sup>2</sup> / (4G)<br>
                &delta;<sup>X</sup><sub>m</sub> = &Delta;E<sub>X</sub>(bulk, m)
                  + &Sigma;<sub>j</sub> x<sub>j</sub> &middot; &Delta;E<sub>X</sub>(j, m)
                  &nbsp;&nbsp;&nbsp; X &isin; {{force, stress}}, &nbsp; &delta;<sub>m</sub> = &delta;<sup>force</sup><sub>m</sub> + &delta;<sup>stress</sup><sub>m</sub><br>
                m* = argmax<sub>m</sub> &delta;<sub>m</sub>:
                &nbsp; &Delta;E<sub>force</sub> = &delta;<sup>force</sup><sub>m*</sub>,
                &nbsp; &Delta;E<sub>stress</sub> = &delta;<sup>stress</sup><sub>m*</sub>,
                &nbsp; &Delta;E<sub>total</sub> = &Delta;E<sub>force</sub> + &Delta;E<sub>stress</sub><br>
                limit = max(&sigma;(&Delta;E<sub>d</sub>),
                {floor_txt} meV/atom)
              </div>
              <p class="text-sm text-slate-600">
                s = structure, m = committee model. &lang;|F<sub>i</sub>|<sup>2</sup>&rang;: mean squared force
                over the atoms (eV<sup>2</sup>/&Aring;<sup>2</sup>); V<sub>atom</sub>: volume per atom
                (&Aring;<sup>3</sup>); &sigma;<sub>dev</sub> = &sigma; &minus; (tr&nbsp;&sigma;/3)&middot;I, with
                |&sigma;<sub>dev</sub>|<sup>2</sup> = &Sigma;<sub>ab</sub> &sigma;<sub>dev,ab</sub><sup>2</sup>
                (GPa<sup>2</sup>; 1 GPa&middot;&Aring;<sup>3</sup> = 6.24 meV); j runs over the phases the bulk is
                compared against on model m's hull (its decomposition products, or the next-best competitor for a
                hull phase) with atom fractions x<sub>j</sub>;
                k = {_fmt(meta.get('force_constant'), 1)} eV/&Aring;<sup>2</sup> (effective force constant),
                G = {_fmt(meta.get('shear_modulus'), 0)} GPa (shear modulus). Both parts are taken from the same
                model m*, so they add up to &Delta;E<sub>total</sub>.
              </p>
              {notes_html}
            </div>
            <div class="overflow-x-auto border rounded-lg">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 text-xs">
                  <tr>
                    <th rowspan="2" class="px-4 py-3 align-bottom">Bulk</th>
                    {head_cells}
                    <th rowspan="2" class="px-4 py-3 align-bottom"><span class="normal-case">&sigma;(&Delta;E<sub>d</sub>)</span></th>
                    <th rowspan="2" class="px-4 py-3 align-bottom">Bias</th>
                    <th rowspan="2" class="px-4 py-3 align-bottom">Votes</th>
                    <th colspan="3" class="px-4 pt-3 pb-1 text-center border-b border-slate-200">Relaxation sensitivity (meV/atom)</th>
                    <th rowspan="2" class="px-4 py-3 align-bottom">Label</th>
                  </tr>
                  <tr>
                    <th class="px-4 pt-1 pb-3">&Delta;E<sub>force</sub></th>
                    <th class="px-4 pt-1 pb-3">&Delta;E<sub>stress</sub></th>
                    <th class="px-4 pt-1 pb-3">&Delta;E<sub>total</sub> / limit</th>
                  </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
              </table>
            </div>{column_notes}
          </div>
        </section>"""


_GREEN_CHIP = "bg-green-100 text-green-700"
_AMBER_CHIP = "bg-amber-100 text-amber-700"
_PURPLE_CHIP = "bg-purple-100 text-purple-700"
_SLATE_CHIP = "bg-slate-100 text-slate-600"
_GEOMETRY_SOURCE_NAMES = {"slab": "slab", "bulk": "bulk", "adsorbate": "adsorbate", "gas": "gas reference"}


def _chip(css, text, title, esc):
    return (f'<span class="inline-block whitespace-nowrap px-2 py-0.5 rounded {css} text-xs '
            f'font-semibold" title="{esc(title)}">{text}</span>')


def _yes_no(ok, title=None, esc=None):
    css, text = ("text-green-700", "yes") if ok else ("text-amber-700", "no")
    attr = f' title="{esc(title)}"' if title and esc else ""
    return f'<span class="{css} font-semibold"{attr}>{text}</span>'


def _geometry_chip(unc, esc, kind):
    """Geometry-sensitivity chip of a facet (``kind="surface"``) or reaction
    candidate (``kind="adsorption"``): a heuristic from the committee models'
    residual forces on the primary's geometries; ``""`` if not computed."""
    if not unc or unc.get("geometry_shift_max") is None:
        return ""
    title = _geometry_title(unc, kind)
    if unc.get("geometry_disagreement"):
        source = _GEOMETRY_SOURCE_NAMES.get(unc.get("geometry_source"), "")
        return _chip(_PURPLE_CHIP, "geometry: sensitive" + (f" ({source})" if source else ""), title, esc)
    return _chip(_SLATE_CHIP, "geometry: low", title, esc)


def _geometry_title(unc, kind):
    """Tooltip of the geometry heuristic: estimate, limit and dominant structure."""
    unit, digits = ("eV/Å²", 4) if kind == "surface" else ("V", 3)
    shift, limit = unc["geometry_shift_max"], unc.get("geometry_limit") or 0.0
    title = (f"Heuristic indicator, not a predicted relaxation energy: the residual forces under "
             f"{unc.get('geometry_shift_model')} on the primary-relaxed geometries correspond to a "
             f"harmonic estimate of {shift:.{digits}f} {unit}, compared with "
             f"max(model spread, floor) = {limit:.{digits}f} {unit}.")
    if unc.get("geometry_disagreement"):
        via = _committee_via_text(unc, kind)
        if via:
            title += f" Dominated by {via}."
        title += " The model spread alone may not reflect the disagreement after relaxation."
    return title


def _geometry_warning(titles, esc):
    """Purple warning marker placed next to a spread whose geometry heuristic
    is flagged; ``""`` when nothing is flagged."""
    if not titles:
        return ""
    return (f' <span class="text-purple-700 font-bold cursor-help" '
            f'title="{esc("Geometry sensitive -- " + chr(10).join(titles))}">&#9888;&#xFE0E;</span>')


def _rank_chip(unc, esc, methods):
    """Facet-rank agreement chip (detailed surface table)."""
    ranks = " / ".join(f"{m} {unc['rank'].get(m, '?')}" for m in methods)
    if unc.get("rank_agree"):
        return _chip(_GREEN_CHIP, "rank: same", f"same facet rank within the bulk under every MLIP ({ranks})", esc)
    return _chip(_AMBER_CHIP, "rank: differs", f"the MLIPs rank this facet differently within the bulk ({ranks})", esc)


def _pds_title(unc, methods):
    return "potential-determining step: " + " / ".join(
        f"{m} step {unc['pds'][m] + 1}" if unc["pds"].get(m) is not None else f"{m} ?" for m in methods)


def _energetic_chips(unc, esc, methods):
    """sigma_eta-tolerance and potential-determining-step chips of a candidate."""
    tol = (unc.get("meta") or {}).get("eta_tolerance")
    chips = []
    if tol is not None:
        ok = unc["eta_std"] <= tol + 1e-12
        chips.append(_chip(_GREEN_CHIP if ok else _AMBER_CHIP,
                           f"&sigma;<sub>&eta;</sub> {'&le;' if ok else '&gt;'} {tol:.2f} V",
                           f"spread of eta over all MLIPs: {unc['eta_std']:.3f} V (tolerance {tol:.2f} V)", esc))
    if unc.get("pds"):
        ok = len(set(unc["pds"].values())) == 1
        chips.append(_chip(_GREEN_CHIP if ok else _AMBER_CHIP, "PDS: same" if ok else "PDS: differs",
                           _pds_title(unc, methods), esc))
    return chips


def _committee_via_text(unc, kind):
    """Structure dominating a flagged geometry shift: ``"*COOH (UMA)"`` for a
    candidate, ``"slab (UMA)"`` / ``"bulk (UMA)"`` for a surface."""
    if kind == "surface":
        model = unc.get("geometry_shift_model")
        return f'{unc.get("geometry_source")} ({model})' if model else ""
    via = unc.get("geometry_via") or {}
    return f'{via.get("species")} ({via.get("model")})' if via.get("model") else ""


def _committee_methods(unc, meta):
    """Primary first, then the committee in config order, restricted to the
    models present in this result."""
    order = [meta["primary"]] + [m["model"] for m in meta.get("committee", [])]
    present = set(unc.get("gamma") or unc.get("eta") or {})
    return [m for m in order if m in present]


def _gamma_cell(surf, meta, esc):
    """Surface energy cell: primary gamma, &plusmn; std over all MLIPs, per-model tooltip."""
    fe = surf["formation_energy"]
    if fe is None:
        return "&mdash;"
    unc = surf.get("uncertainty")
    if not unc or not meta:
        return f"{fe:.4f}"
    per_model = " · ".join(f'{m} {unc["gamma"][m]:.4f}' for m in _committee_methods(unc, meta))
    return (f'<span title="{esc(per_model + " (eV/Å²)")}">{fe:.4f} &plusmn; '
            f'{unc["gamma_std"]:.4f}</span>')


def _eta_cell(best, esc):
    """Best-eta cell: primary eta, &plusmn; std over all MLIPs, and the three
    separate checks (sigma_eta tolerance, potential-determining step, geometry)."""
    if not best:
        return "&mdash;"
    unc = best.get("uncertainty")
    if not unc:
        return f'{best["eta"]:.3f} V'
    meta = unc.get("meta") or {}
    methods = _committee_methods(unc, meta) if meta else list(unc["eta"])
    per_model = " · ".join(f'{m} {unc["eta"][m]:.3f}' for m in methods)
    chips = _energetic_chips(unc, esc, methods) + [_geometry_chip(unc, esc, "adsorption")]
    return (f'<span class="whitespace-nowrap" title="{esc(per_model + " (V)")}">{best["eta"]:.3f} &plusmn; '
            f'{unc["eta_std"]:.3f} V</span>'
            f'<div class="flex flex-wrap gap-1 mt-1">{"".join(c for c in chips if c)}</div>')


def _surface_name(summary, surface_id):
    for surf in summary["surfaces"]:
        if str(surf["surface_id"]) == str(surface_id):
            miller = tuple(surf["miller_index"]) if surf["miller_index"] else "?"
            return f"{surface_id} {miller}"
    return str(surface_id)


def _candidate_name(summary, row_id):
    """Full identity of an adsorption candidate -- the unit the best-candidate
    agreement compares: ``"#15 · surface 1 (1, 0, 0) · ontop · 0.5 ML"``."""
    for c in summary["reaction_candidates"]:
        if str(c["row_id"]) == str(row_id):
            miller = tuple(c["miller_index"]) if c["miller_index"] else "?"
            bits = [f"#{row_id}", f"surface {c['surface_id']} {miller}", c["site_type"]]
            if c.get("coverage") is not None:
                bits.append(f"{c['coverage']:g} ML")
            return " · ".join(bits)
    return f"#{row_id}"


def _agreement_bit(agree, picks, primary, name_fn):
    """"same under all MLIPs" or the models that pick something else."""
    if agree:
        return '<span class="text-green-700 font-semibold">same under all MLIPs</span>'
    others = [f"{m} &rarr; {name_fn(v)}" for m, v in picks.items() if v != picks.get(primary)]
    return ('<span class="text-amber-700 font-semibold">differs</span> '
            f'(primary {name_fn(picks.get(primary))}; {"; ".join(others)})')


def _committee_note(summary, esc):
    """One line under a bulk's surfaces table: does the committee agree on the
    lowest-gamma facet, the facets carried to adsorption and the best candidate,
    and how far apart are the models on gamma?"""
    bits = []
    surf_bulk, surf_meta = summary.get("surface_uncertainty"), summary.get("surface_uncertainty_meta")
    if surf_bulk and surf_meta:
        name = lambda sid: esc(_surface_name(summary, sid))
        bits.append("Lowest-&gamma; facet: " + _agreement_bit(
            surf_bulk["top1_agree"], surf_bulk["top1"], surf_meta["primary"], name))
        bits.append(f"Facets carried to adsorption (lowest {surf_meta.get('n_selected')}): " + (
            '<span class="text-green-700 font-semibold">same under all MLIPs</span>'
            if surf_bulk["selected_agree"] else '<span class="text-amber-700 font-semibold">differ</span>'))
        gammas = [surf["uncertainty"]["gamma_std"] for surf in summary["surfaces"] if surf.get("uncertainty")]
        if gammas:
            bits.append(f"&sigma;<sub>&gamma;</sub> over facets: {min(gammas):.4f}&ndash;{max(gammas):.4f} eV/&Aring;&sup2;")
    ads_bulk, ads_meta = summary.get("adsorption_uncertainty"), summary.get("adsorption_uncertainty_meta")
    if ads_bulk and ads_meta:
        name = lambda rid: esc(_candidate_name(summary, rid))
        bits.append("Best adsorption candidate: " + _agreement_bit(
            ads_bulk["best_agree"], ads_bulk["best"], ads_meta["primary"], name))
    if not bits:
        return ""
    return ('<p class="text-sm text-slate-600 -mt-4 mb-6">'
            + " &nbsp;|&nbsp; ".join(bits) + "</p>")


def _surface_adsorption_section(summaries, esc):
    """"Surface &amp; Adsorption Uncertainty" section: one row per bulk, with
    ranking stability, energetic agreement and geometry sensitivity reported as
    separate columns (no combined label). ``""`` when neither analysis ran."""
    rows_with = [s for s in summaries if s.get("surface_uncertainty") or s.get("adsorption_uncertainty")]
    if not rows_with:
        return ""
    surf_meta = next((s["surface_uncertainty_meta"] for s in summaries if s.get("surface_uncertainty_meta")), None)
    ads_meta = next((s["adsorption_uncertainty_meta"] for s in summaries if s.get("adsorption_uncertainty_meta")), None)
    meta = surf_meta or ads_meta
    primary = meta["primary"]
    eta_tol = ads_meta.get("eta_tolerance") if ads_meta else None
    dash = "&mdash;"

    rows = []
    for s in rows_with:
        uid = s["structure_uuid"]
        surf_bulk = s.get("surface_uncertainty")
        top1 = _yes_no(surf_bulk["top1_agree"]) if surf_bulk else dash
        selected = _yes_no(surf_bulk["selected_agree"]) if surf_bulk else dash

        gammas = [surf["uncertainty"]["gamma_std"] for surf in s["surfaces"] if surf.get("uncertainty")]
        facet_flags = [f"facet {_surface_name(s, surf['surface_id'])}: {_geometry_title(surf['uncertainty'], 'surface')}"
                       for surf in s["surfaces"]
                       if surf.get("uncertainty") and surf["uncertainty"].get("geometry_disagreement")]
        gamma_cell = (f'<span title="{esc(f"largest of the {len(gammas)} per-facet spreads; range {min(gammas):.4f}–{max(gammas):.4f} eV/Å²")}">'
                      f'{max(gammas):.4f}</span>{_geometry_warning(facet_flags, esc)}' if gammas else dash)

        ads_bulk = s.get("adsorption_uncertainty")
        if ads_bulk:
            if ads_bulk["best_agree"]:
                best_agree = _yes_no(True, f"every MLIP picks {_candidate_name(s, ads_bulk['best'][primary])}", esc)
            else:
                picks = ads_bulk["best"]
                others = [f"{m} &rarr; {esc(_candidate_name(s, v))}" for m, v in picks.items()
                          if v != picks.get(primary)]
                best_agree = (_yes_no(False) + '<div class="text-xs text-slate-500 mt-1 min-w-[14rem]">'
                              + "<br>".join(others) + "</div>")
        else:
            best_agree = dash

        best = s.get("best_candidate")
        best_unc = best.get("uncertainty") if best else None
        if best_unc and ads_meta:
            methods = _committee_methods(best_unc, ads_meta)
            eta_cell = (f'<span class="whitespace-nowrap" title="{esc(_candidate_name(s, best["row_id"]))}">'
                        + " / ".join(f'{best_unc["eta"][m]:.2f}' for m in methods) + "</span>")
            over = eta_tol is not None and best_unc["eta_std"] > eta_tol + 1e-12
            std_title = (f"sample standard deviation (N-1) of the {len(methods)} eta values "
                         + ", ".join(f'{m} {best_unc["eta"][m]:.3f}' for m in methods)
                         + f" V = {best_unc['eta_std']:.3f} V"
                         + (f"; {'above' if over else 'within'} the {eta_tol:.2f} V tolerance" if eta_tol is not None else ""))
            eta_flags = ([_geometry_title(best_unc, "adsorption")]
                         if best_unc.get("geometry_disagreement") else [])
            eta_std = (f'<span class="{"text-amber-700 font-semibold" if over else "text-green-700"}" '
                       f'title="{esc(std_title)}">{best_unc["eta_std"]:.3f}</span>'
                       f'{_geometry_warning(eta_flags, esc)}')
        else:
            eta_cell = eta_std = dash

        rows.append(f"""
                  <tr class="border-t align-top">
                    <td class="px-4 py-3 font-mono text-xs" title="{esc(uid)}">{esc(uid[:8])}</td>
                    <td class="px-4 py-3 border-l border-slate-200">{top1}</td>
                    <td class="px-4 py-3">{selected}</td>
                    <td class="px-4 py-3">{best_agree}</td>
                    <td class="px-4 py-3 border-l border-slate-200 text-right">{gamma_cell}</td>
                    <td class="px-4 py-3">{eta_cell}</td>
                    <td class="px-4 py-3 text-right whitespace-nowrap">{eta_std}</td>
                  </tr>""")

    order = " / ".join([primary] + [m["model"] for m in meta.get("committee", [])])
    committee_bit = ", ".join(
        f'{esc(m["model"])}' + (f' ({esc(m["head"])})' if m.get("head") else "")
        for m in meta.get("committee", []))
    k = _fmt(meta.get("force_constant"), 1)
    gamma_floor = _fmt(surf_meta.get("gamma_floor") if surf_meta else None, 4)
    eta_floor = _fmt(ads_meta.get("eta_floor") if ads_meta else None, 2)
    eta_tol_txt = _fmt(eta_tol, 2)
    n_selected = surf_meta.get("n_selected") if surf_meta else None
    n_models = 1 + len(meta.get("committee", []))
    p = esc(primary)

    return f"""
        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="activity" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Surface &amp; Adsorption Uncertainty</h2>
          </div>
          <div class="bg-white border rounded-xl p-6">
            <div class="border-l-4 border-primary bg-blue-50 text-slate-700 text-sm rounded-r-lg p-4 mb-5 space-y-3">
              <div>
                <h3 class="font-semibold text-slate-900 mb-1">What was compared</h3>
                <p>
                  <strong>{p}</strong> is used to generate and relax the bulk and surface structures, identify
                  adsorption candidates, and select the best reaction candidate. {committee_bit} then evaluate the
                  same {p}-relaxed slabs, adsorbate systems and gas references by single points, and each model
                  computes &gamma; and &eta; on its own energies.
                  The table asks whether the screening
                  conclusions would change under another model. The spread measures disagreement between
                  MLIPs; it is not an error bar calibrated against DFT.
                  We compare the models in three ways, reported separately on purpose (a row can agree on
                  energies and still change the ranking, or vice versa): <em>ranking stability</em> and
                  <em>energetic agreement</em> (the table columns, explained below it) and <em>geometry
                  sensitivity</em> (below). The per-bulk tables further down give every facet and candidate, and
                  the reaction-path diagrams overlay each committee model as thin lines.
                </p>
              </div>
              <div class="bg-white border rounded-lg p-3 text-sm text-slate-600 space-y-1">
                <h4 class="font-semibold text-slate-900 text-sm">Geometry sensitivity <span class="font-normal text-slate-500">(heuristic)</span></h4>
                <p>
                  The committee models evaluate {p}'s geometries, not their own minima, so their energies could change
                  if they relaxed the structures themselves. Residual forces give a rough harmonic estimate of that
                  effect, which is compared with max(model spread, floor); the floor is {eta_floor} V for &eta; and
                  {gamma_floor} eV/&Aring;&sup2; for &gamma;. It is an indicator, not a predicted relaxation energy.
                </p>
                <ul class="list-disc pl-5 space-y-1">
                  <li><span class="text-purple-700 font-bold">&#9888;&#xFE0E;</span> next to &sigma;<sub>&eta;</sub> &mdash; the best
                    candidate is flagged; next to Max &sigma;<sub>&gamma;</sub> &mdash; at least one facet of the bulk is
                    flagged. For at least one committee model the estimate exceeds the limit: that model is far from its
                    own minimum on the {p} geometry, and relaxing could move its value by as much as the model spread or
                    more. The spread is then not a reliable measure of disagreement on its own; after relaxation it could
                    be larger or smaller. Hover over the marker for the estimate, the limit and the dominant structure.</li>
                  <li>No marker &mdash; the estimate is below the limit: the residual forces are small compared with how
                    much the models already disagree, so the spread can be read at face value.</li>
                </ul>
                <p>
                  In the per-bulk tables below the same check is shown for every facet (&gamma; checks) and for the best
                  candidate of every facet (next to its &eta;) as a chip:
                  <span class="px-1.5 rounded bg-slate-100 text-slate-600 font-semibold">geometry: low</span> or
                  <span class="px-1.5 rounded bg-purple-100 text-purple-700 font-semibold">geometry: sensitive (&hellip;)</span>.
                  The word in brackets names the structure that dominates the estimate:
                  <em>adsorbate</em> = an adsorbed intermediate (*O, *OH, *OOH, &hellip;);
                  <em>slab</em> = the clean slab *;
                  <em>gas reference</em> = a gas molecule (H<sub>2</sub>O, H<sub>2</sub>, O<sub>2</sub>);
                  <em>bulk</em> = the bulk reference in &gamma; (facets only). The tooltip names the exact species and
                  model, e.g. &ldquo;dominated by *OOH (UMA)&rdquo;.
                </p>
              </div>
              <details class="text-sm text-slate-600">
                <summary class="cursor-pointer font-semibold text-slate-700">Definitions and thresholds</summary>
                <div class="bg-white border rounded-lg px-4 py-3 mt-2 font-mono leading-6 overflow-x-auto">
                  &gamma;<sub>m</sub> = (E<sub>slab,m</sub> &minus; N<sub>slab</sub> &middot; &epsilon;<sub>bulk,m</sub>) / (2A)<br>
                  &eta;<sub>m</sub>, &Delta;G<sub>i,m</sub> = the reaction's CHE function on model m's energies
                  (same ZPE, references and pinning)<br>
                  &sigma;<sub>&eta;</sub> = &radic;[ &Sigma;<sub>m</sub> (&eta;<sub>m</sub> &minus; &eta;&#772;)<sup>2</sup> / (N &minus; 1) ],
                    &eta;&#772; = &Sigma;<sub>m</sub> &eta;<sub>m</sub> / N, m &isin; {{{esc(order)}}}, for one fixed candidate
                    ({p}'s best); &sigma;<sub>&gamma;</sub> likewise per facet<br>
                  heuristic geometry estimates (harmonic, force constant k):<br>
                  &nbsp;&nbsp;&delta;&gamma;<sub>m</sub> = [ &Sigma;<sub>i</sub>|F<sub>i</sub>|<sup>2</sup><sub>slab</sub> / (2k)
                    + N<sub>slab</sub> &middot; &lang;|F|<sup>2</sup>&rang;<sub>bulk</sub> / (2k) ] / (2A)<br>
                  &nbsp;&nbsp;&delta;&eta;<sub>m</sub> = max<sub>i</sub> &Sigma;<sub>s</sub> |&part;&Delta;G<sub>i</sub>/&part;E<sub>s</sub>|
                    &middot; &Sigma;<sub>j</sub>|F<sub>j</sub>|<sup>2</sup><sub>s</sub> / (2k)<br>
                  sensitive: max<sub>m</sub> &delta;&gamma;<sub>m</sub> &gt; max(&sigma;<sub>&gamma;</sub>, {gamma_floor} eV/&Aring;<sup>2</sup>)
                  &nbsp;or&nbsp; max<sub>m</sub> &delta;&eta;<sub>m</sub> &gt; max(&sigma;<sub>&eta;</sub>, {eta_floor} V)
                </div>
                <p class="mt-2">
                  &epsilon;<sub>bulk,m</sub>: model m's energy per atom of the bulk the slab was cut from;
                  A: slab area; s: species of the candidate (clean slab *, intermediates, gas references per
                  molecule); &part;&Delta;G<sub>i</sub>/&part;E<sub>s</sub>: coefficient of E<sub>s</sub> in step i;
                  forces of fixed atoms are excluded; stress is not used (slab cells are never relaxed);
                  k = {k} eV/&Aring;<sup>2</sup>. The estimate assumes an isotropic harmonic well with a single
                  force constant, so it indicates where geometry may matter rather than predicting the energy
                  change.
                </p>
              </details>
            </div>
            <div class="overflow-x-auto border rounded-lg">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 text-xs">
                  <tr>
                    <th class="px-4 pt-3 pb-1"></th>
                    <th colspan="3" class="px-4 pt-3 pb-1 text-center border-b border-l border-slate-200">Ranking stability</th>
                    <th colspan="3" class="px-4 pt-3 pb-1 text-center border-b border-l border-slate-200">Energetic agreement</th>
                  </tr>
                  <tr>
                    <th class="px-4 py-3">Bulk</th>
                    <th class="px-4 py-3 border-l border-slate-200" title="All MLIPs pick the same lowest-gamma facet">Lowest facet agrees</th>
                    <th class="px-4 py-3" title="All MLIPs pick the same facets for adsorption">Adsorption facets agree</th>
                    <th class="px-4 py-3" title="All MLIPs pick the same lowest-eta adsorption candidate (facet + site + coverage)">Best candidate agrees</th>
                    <th class="px-4 py-3 border-l border-slate-200 text-right" title="largest per-facet standard deviation of gamma over all models">Max <span class="normal-case">&sigma;<sub>&gamma;</sub></span> (eV/&Aring;&sup2;)</th>
                    <th class="px-4 py-3" title="eta of the primary's best candidate under each model">Best &eta; <span class="normal-case">({esc(order)})</span> (V)</th>
                    <th class="px-4 py-3 text-right" title="sample standard deviation of the Best-eta values over all models"><span class="normal-case">&sigma;<sub>&eta;</sub></span> (V)</th>
                  </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
              </table>
            </div>
            <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600 mt-5">
              <div>
                <dt class="font-semibold text-slate-900">Bulk</dt>
                <dd>Bulk structure id (first 8 characters). Hover the pointer over an id to see the full UUID.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Ranking stability</dt>
                <dd>Would another model make a different decision in the workflow? The next three columns say
                  <em>yes</em> when every model makes the same choice as {p}.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Lowest facet agrees</dt>
                <dd>Every model picks the same lowest-&gamma; facet.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Adsorption facets agree</dt>
                <dd>Every model picks the same set of facets to carry to adsorption (lowest {_fmt(n_selected, 0)} by &gamma;).</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Best candidate agrees</dt>
                <dd>Every model picks the same lowest-&eta; adsorption candidate, compared as the complete
                  configuration (facet, site, coverage; shown as #id), not just the facet. On <em>no</em>, the models
                  that pick a different candidate are listed under it; hover the pointer over a <em>yes</em> to see
                  the shared candidate.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Energetic agreement</dt>
                <dd>How far apart are the numbers? The last three columns.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Max &sigma;<sub>&gamma;</sub> (eV/&Aring;&sup2;)</dt>
                <dd>&sigma;<sub>&gamma;</sub> is the sample standard deviation of a facet's surface energy over all
                  models; the largest over the bulk's facets is shown (per-facet values are in the per-bulk tables).
                  Hover the pointer over a value to see the range over facets. &#9888;&#xFE0E; marks a geometry-sensitive
                  facet (see <em>Geometry sensitivity</em> above).</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">Best &eta; ({esc(order)}) (V)</dt>
                <dd>Overpotential of {p}'s best candidate, computed by each model on its own energies (one
                  fixed candidate, not each model's own best). Hover the pointer over a value to see which candidate
                  it is.</dd>
              </div>
              <div>
                <dt class="font-semibold text-slate-900">&sigma;<sub>&eta;</sub> (V)</dt>
                <dd>Sample standard deviation of the Best &eta; values (N = {n_models} models, {p} included); green if
                  &le; {eta_tol_txt} V, amber above. Hover the pointer over a value to see the per-model values.
                  &#9888;&#xFE0E; marks a geometry-sensitive best candidate.</dd>
              </div>
            </dl>
          </div>
        </section>"""


def _keep_unit_case(html_doc):
    """Table headers and the small metadata labels are styled ``uppercase``,
    which would print "eV" as "EV", "eta" as a capital Eta and
    "E<sub>hull</sub>" as "E<sub>HULL</sub>". Inside every ``<th>`` and every
    ``uppercase``-classed ``<p>``, wrap parenthesised units, ``&eta;`` and
    lower-case subscripts in a ``normal-case`` span."""
    import re

    def fix(m):
        inner = m.group(2)
        inner = re.sub(r"(?<=\s)(\([^()]*\))", r'<span class="normal-case">\1</span>', inner)
        inner = inner.replace("&eta;", '<span class="normal-case">&eta;</span>')
        inner = re.sub(r"(<sub>[a-z][^<]*</sub>)", r'<span class="normal-case">\1</span>', inner)
        return m.group(1) + inner + m.group(3)

    html_doc = re.sub(r"(<th\b[^>]*>)(.*?)(</th>)", fix, html_doc, flags=re.S)
    return re.sub(r'(<p class="[^"]*\buppercase\b[^"]*">)(.*?)(</p>)', fix, html_doc, flags=re.S)


def render_html_report(chemical_formula, reaction, reaction_path, summaries=None,
                        output_path="report.html"):
    """Render a self-contained HTML report (layout mirrors ``result-sample.html``)
    for one (composition, reaction, reaction_path): a table of stable bulks
    (uuid + provenance source + MP id for MPDB-sourced bulks), one surfaces
    table per bulk (miller index,
    surface formation energy, size), and one reaction-path free-energy
    diagram per surface. Figures are embedded as base64 PNGs, so the output
    file has no external dependencies besides the Tailwind/lucide CDN
    scripts used for layout/icons.

    Also writes ``raw_data.json`` next to ``output_path`` (via ``raw_data()``)
    and links a "Download Raw Data (JSON)" button to it -- every number and
    structure behind the page, not just what's rendered into the tables.

    ``summaries`` lets a caller reuse an already-computed ``summarize()`` (or
    ``report()``) result instead of re-querying the database; otherwise
    ``summarize()`` is called here. Writes to ``output_path`` and also
    returns the HTML string."""
    import os
    import json
    import html as _html

    if summaries is None:
        summaries = summarize(chemical_formula, reaction, reaction_path)

    raw_data_filename = "raw_data.json"

    n_bulks = len(summaries)
    unc_present = any(s["bulk"] and s["bulk"].get("uncertainty") for s in summaries)
    n_robust = sum(1 for s in summaries
                   if s["bulk"] and (s["bulk"].get("uncertainty") or {}).get("label") == "robust")
    n_unc = sum(1 for s in summaries if s["bulk"] and s["bulk"].get("uncertainty"))
    n_surfaces = sum(s["n_surfaces"] for s in summaries)
    n_candidates = sum(s["n_reaction_candidates"] for s in summaries)
    if n_candidates == 0:
        reasons = "".join(f"<li>{_html.escape(r)}</li>" for r in no_candidate_reasons(summaries))
        no_candidate_note = f"""
            <div class="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <p class="font-bold mb-2">No reaction-path candidate was found for this pathway.</p>
              <ul class="list-disc pl-5 space-y-1">{reasons}</ul>
              <p class="mt-2 text-amber-800">The per-surface outcome is in the <em>Adsorbate screen</em>
                column of the per-bulk tables below.</p>
            </div>"""
    else:
        no_candidate_note = ""
    gaps = hull_gaps(chemical_formula)
    if gaps:
        gap_items = "".join(f"<li>{_html.escape(line)}</li>" for line in hull_gap_lines(gaps))
        hull_gap_note = f"""
            <div class="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <p class="font-bold mb-2">The phase diagram has incomplete competing phases.</p>
              <ul class="list-disc pl-5 space-y-1">{gap_items}</ul>
              <p class="mt-2 text-amber-800">A stable competing phase missing from these subsystems would
                place the hull too high, so the E<sub>hull</sub> values below may be underestimated.</p>
            </div>"""
    else:
        hull_gap_note = ""
    model =next((s["bulk"]["ml_bulk_model"] for s in summaries if s["bulk"]), None)
    bulk_task = _ml_stage_head("bulk_relax")
    surface_model = _ml_surface_model()
    surface_task = _ml_stage_head("face_build")

    # -- light-harvesting screen (OpticalScreenWorkChain) --------------------
    electronic_present = any(s.get("electronic") for s in summaries)
    couple = _reaction_couple(reaction, reaction_path)
    n_light = 0
    light_partial = []  # (uuid, [passing model keys]) for model-dependent verdicts
    screen_models = screen_fidelity = None
    screen_targets = []
    for s in summaries:
        e = s.get("electronic")
        if not e:
            continue
        if screen_models is None:
            screen_models = ", ".join(e.get("gap_models") or []) or None
            screen_fidelity = e.get("megnet_fidelity_label")
            screen_targets = _gap_model_targets(e)
        consensus, passing = straddle_consensus(bulk_straddle_by_model(e, reaction, reaction_path))
        if consensus == "straddles":
            n_light += 1
        elif consensus == "model-dependent":
            light_partial.append((s["structure_uuid"], passing))

    # short display name per gap model ("MEGNet", "ALIGNN"); with the target
    # level added when two models would otherwise share a name
    model_keys = [k for k, _, _, _ in screen_targets]
    model_names = {k: (m or k).split()[0] for k, m, _, _ in screen_targets}
    if len(set(model_names.values())) < len(model_names):
        model_names = {k: f"{(m or k).split()[0]} ({t})" if t else k for k, m, _, t in screen_targets}
    multi_model = len(model_keys) > 1

    def esc(x):
        return _html.escape(str(x)) if x is not None else "&mdash;"

    def mp_id_cell(bulk):
        """Materials Project id for a bulk whose provenance is the MPDB, as a
        link to its MP page; ``&mdash;`` for any other source (csp/generated/
        ...) or when no id was stored."""
        if not bulk:
            return "&mdash;"
        source, mp_id = bulk.get("source"), bulk.get("mp_id")
        if not mp_id or not (source or "").startswith("MPDB"):
            return "&mdash;"
        return (f'<a href="https://materialsproject.org/materials/{esc(mp_id)}" '
                f'target="_blank" rel="noopener" class="text-primary hover:underline '
                f'font-mono text-xs">{esc(mp_id)}</a>')

    def straddle_badge(by_model):
        """Three-way verdict over the gap models; the hover lists each model's margins."""
        consensus, passing = straddle_consensus(by_model)
        if consensus is None:
            return "&mdash;"
        order = sorted(by_model, key=lambda k: model_keys.index(k) if k in model_keys else len(model_keys))
        passing = [k for k in order if k in passing]
        colour, label = {
            "straddles": ("bg-green-100 text-green-700", "straddles"),
            "model-dependent": ("bg-amber-100 text-amber-800",
                                f'model-dependent ({", ".join(model_names.get(k, k) for k in passing)} only)'),
            "no": ("bg-red-100 text-red-700", "no"),
        }[consensus]
        title = "; ".join(f'{model_names.get(k, k)}: reduction margin {by_model[k]["margin_reduction_V"]:+.4f} V, '
                          f'oxidation margin {by_model[k]["margin_oxidation_V"]:+.4f} V' for k in order)
        return (f'<span class="px-2 py-0.5 rounded {colour} text-xs font-bold" '
                f'title="{esc(title)}">{esc(label)}</span>')

    def per_model_cell(values, fmt, below=None):
        """``a / b`` in gap-model order; values under ``below`` in red."""
        parts = []
        for k in model_keys:
            v = values.get(k)
            if v is None:
                parts.append("&mdash;")
                continue
            txt = format(v, fmt).replace("-", "&minus;")
            parts.append(f'<span class="text-red-700">{txt}</span>' if below is not None and v < below else txt)
        return f'<span class="whitespace-nowrap">{" / ".join(parts)}</span>'

    # -- stable-bulks table ----------------------------------------------
    bulk_rows = []
    for s in summaries:
        b, uid = s["bulk"], s["structure_uuid"]
        if b:
            ehull_cell, source_cell = _ehull_with_spread(b, esc), esc(_source_label(b["source"]))
        else:
            ehull_cell = source_cell = "&mdash;"
        bulk_rows.append(f"""
                  <tr class="border-t">
                    <td class="px-6 py-4 font-mono text-xs text-slate-900" title="{esc(uid)}">{esc(uid[:8])}</td>
                    <td class="px-6 py-4">{source_cell}</td>
                    <td class="px-6 py-4">{mp_id_cell(b)}</td>
                    <td class="px-6 py-4">{ehull_cell}</td>
                    <td class="px-6 py-4">{s["n_surfaces"]}</td>
                  </tr>""")

    # -- bulk-table notes: model names come from this run's data --------------
    unc_meta = next((s["bulk"]["uncertainty_meta"] for s in summaries
                     if s["bulk"] and s["bulk"].get("uncertainty_meta")), None)
    if unc_meta:
        ehull_models = _uncertainty_methods(unc_meta)
        ehull_note = (f"The central value is the {esc(ehull_models[0])} prediction. The &plusmn; value "
                      f"is the standard deviation across {esc(_join_names(ehull_models))}. It represents "
                      "disagreement between these models, not an uncertainty interval calibrated "
                      "against DFT. Hover the pointer over a value to see the per-model values "
                      f"({esc(', '.join(ehull_models))}) and their range.")
    else:
        by_model = f" by {esc(model)}" if model else ""
        ehull_note = (f"ML-predicted energy above the convex hull{by_model}; no committee "
                      "spread is available for this run.")
    bulk_notes = f"""
              <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600">
                <div>
                  <dt class="font-semibold text-slate-900">UUID</dt>
                  <dd>Unique identifier of the bulk structure in the uvsib structure database. Only the
                    first 8 characters are shown; hover the pointer over an id to see the full UUID. The same identifier
                    labels this bulk throughout the report.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">MP ID</dt>
                  <dd>Materials Project identifier of the structure, linked to its Materials Project
                    page. Shown only for structures retrieved from the Materials Project database
                    (<em>MPDB</em> sources); &mdash; for CSP and ML-generated structures.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Source</dt>
                  <dd><em>MPDB(exp.)</em>: experimentally observed structures retrieved from the Materials
                    Project database. <em>MPDB(stb.)</em>: structures retrieved from the Materials Project
                    database as computationally stable. <em>CSP</em>: structures from crystal structure
                    prediction. <em>generated</em>: structures generated by ML models.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">E above hull (eV/atom)</dt>
                  <dd>{ehull_note}</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Surfaces retained</dt>
                  <dd>Number of surface slab models generated and retained for this bulk structure.
                    Different slabs may represent different orientations or surface configurations. A value
                    of zero means that no surface candidates were retained by the workflow; it does not
                    establish that the material has no stable physical surfaces.</dd>
                </div>
              </dl>"""

    # -- per-bulk sections: surfaces table + reaction-path FED grid ------
    bulk_sections = []
    for s in summaries:
        uid, b = s["structure_uuid"], s["bulk"]
        ehull_bit = f'E<sub>hull</sub> = {b["ehull"]:.4f} eV/atom' if b else "E<sub>hull</sub> unknown"
        if b and b.get("uncertainty"):
            unc = b["uncertainty"]
            ehull_bit += (f' (committee {unc["ehull_min"]:.3f}&ndash;{unc["ehull_max"]:.3f}, '
                          f'{_stability_badge(unc, esc)})')
        source_bit = esc(_source_label(b["source"])) if b else "&mdash;"
        mp_id_bit = f' &nbsp;|&nbsp; MP ID: {mp_id_cell(b)}' if (b and mp_id_cell(b) != "&mdash;") else ""

        surface_rows = []
        surf_meta = s.get("surface_uncertainty_meta")
        surf_unc_present = bool(surf_meta) and any(surf.get("uncertainty") for surf in s["surfaces"])
        for surf in s["surfaces"]:
            miller = str(tuple(surf["miller_index"])) if surf["miller_index"] else "?"
            best = surf["best_candidate"]
            eta_cell = _eta_cell(best, esc)
            # surf["n_atoms"]/["area"] are the base (1x1) slab DBSurface
            # stored; the best candidate on this surface may have been
            # computed on a repeated supercell (different candidates on the
            # SAME surface can use different repeats for different
            # coverages), so scale by that candidate's own repeat rather
            # than showing the un-repeated base numbers alongside its eta.
            mult = _repeat_multiplier(best.get("repeat")) if best else 1
            n_atoms_cell = surf["n_atoms"] * mult if surf["n_atoms"] is not None else "&mdash;"
            area_cell = f'{surf["area"] * mult:.2f}' if surf["area"] is not None else "&mdash;"
            fe_cell = _gamma_cell(surf, surf_meta, esc)
            if surf_unc_present:
                unc = surf.get("uncertainty")
                if unc:
                    methods = _committee_methods(unc, surf_meta)
                    rank_cell = (f'<span title="{esc(" / ".join(methods))}">'
                                 + " / ".join(str(unc["rank"].get(m, "?")) for m in methods) + "</span>")
                    label_cell = ('<div class="flex flex-wrap gap-1">' + _rank_chip(unc, esc, methods)
                                  + _geometry_chip(unc, esc, "surface") + "</div>")
                else:
                    rank_cell = label_cell = "&mdash;"
                unc_cells = (f'<td class="px-4 py-3">{rank_cell}</td>'
                             f'<td class="px-4 py-3">{label_cell}</td>')
            else:
                unc_cells = ""
            repeat = best.get("repeat") if best else None
            repeat_cell = f"({repeat[0]}, {repeat[1]})" if repeat and len(repeat) >= 2 else "&mdash;"
            surface_rows.append(f"""
                      <tr class="border-t">
                        <td class="px-4 py-3 font-mono text-xs">{surf["surface_id"]}</td>
                        <td class="px-4 py-3">{esc(miller)}</td>
                        <td class="px-4 py-3">{fe_cell}</td>
                        {unc_cells}
                        <td class="px-4 py-3">{n_atoms_cell}</td>
                        <td class="px-4 py-3">{area_cell}</td>
                        <td class="px-4 py-3">{eta_cell}</td>
                        <td class="px-4 py-3">{repeat_cell}</td>
                        <td class="px-4 py-3">{_adsorbate_screen_cell(surf, esc)}</td>
                      </tr>""")
        surface_rows_html = "".join(surface_rows) or (
            '<tr><td colspan="8" class="px-4 py-4 text-slate-400 italic">'
            'no stable surfaces found</td></tr>')

        if surf_unc_present:
            order = " / ".join([surf_meta["primary"]] + [m["model"] for m in surf_meta.get("committee", [])])
            unc_head = (f'<th class="px-4 py-3" title="Rank of the facet within this bulk ({esc(order)})">'
                        '&gamma; rank</th><th class="px-4 py-3">&gamma; checks</th>')
        else:
            unc_head = ""
        committee_note = _committee_note(s, esc)

        if any(surf["best_candidate"] for surf in s["surfaces"]):
            fed_block = (f'<img src="{_fig_to_data_uri(plot_bulk_detail(s, reaction, reaction_path))}" '
                         f'alt="Reaction path diagrams for bulk {esc(uid[:8])}" '
                         # Capped at the width it had in the old 2/3 results column so the
                         # wider page layout does not upscale the figure.
                         f'class="block mx-auto w-full max-w-[729px] rounded-lg border" />')
        elif s["surfaces"]:
            fed_block = ('<p class="text-sm text-slate-400 italic">'
                         "no reaction-path candidates for this bulk's surfaces</p>")
        else:
            fed_block = '<p class="text-sm text-slate-400 italic">no stable surfaces to plot</p>'

        bulk_sections.append(f"""
        <div class="rounded-xl border bg-white p-6">
          <h3 class="font-bold text-slate-900 mb-1">Bulk {esc(uid[:8])}</h3>
          <p class="text-xs text-slate-400 font-mono mb-3">{esc(uid)}</p>
          <p class="text-sm text-slate-600 mb-4">{ehull_bit} &nbsp;|&nbsp; source: {source_bit}{mp_id_bit}</p>

          <div class="overflow-x-auto mb-6 border rounded-lg">
            <table class="w-full text-sm text-left text-slate-700">
              <thead class="bg-slate-50 text-slate-500 text-xs">
                <tr>
                  <th class="px-4 py-3">Surface ID</th>
                  <th class="px-4 py-3">Miller Index</th>
                  <th class="px-4 py-3">Surface Energy (eV/&Aring;&sup2;)</th>
                  {unc_head}
                  <th class="px-4 py-3" title="Scaled to the best candidate's supercell repeat, not the bare relaxed slab">
                    # Atoms</th>
                  <th class="px-4 py-3" title="Scaled to the best candidate's supercell repeat, not the bare relaxed slab">
                    Area (&Aring;&sup2;)</th>
                  <th class="px-4 py-3">Best &eta;</th>
                  <th class="px-4 py-3" title="In-plane supercell repeat (nx, ny) the best candidate's clean slab was built on, before the adsorbate was placed">
                    Repeat (n<sub>x</sub>, n<sub>y</sub>)</th>
                  <th class="px-4 py-3" title="What happened to the adsorption sites on this surface">Adsorbate screen</th>
                </tr>
              </thead>
              <tbody>{surface_rows_html}</tbody>
            </table>
          </div>
          {committee_note}

          <h4 class="font-bold text-slate-900 mb-2 text-sm">Reaction Path Diagrams</h4>
          {fed_block}
        </div>""")

    # -- per-bulk surface-table terms, explained once under the section title --
    surf_meta_any = next((s.get("surface_uncertainty_meta") for s in summaries
                          if s.get("surface_uncertainty_meta")
                          and any(surf.get("uncertainty") for surf in s["surfaces"])), None)
    surf_primary = (surf_meta_any or {}).get("primary") or surface_model or "the primary model"
    if surf_meta_any:
        surf_order = " / ".join([surf_meta_any["primary"]]
                                + [m["model"] for m in surf_meta_any.get("committee", [])])
        spread_bit = (", with the cross-model spread (&plusmn;, standard deviation). Hover the pointer over a "
                      "value to see the per-model values")
        eta_spread_bit = (", with the spread across models (&plusmn;) for that same candidate; the chips "
                          "show whether the spread is within tolerance, whether the potential-determining "
                          "step (PDS) is the same under every model, and the geometry sensitivity. Hover the "
                          "pointer over a value to see the per-model &eta;, or over a chip to see the spread, "
                          "the per-model PDS, or the geometry-sensitivity estimate")
        unc_terms = f"""
                <div>
                  <dt class="font-semibold text-slate-900">&gamma; rank</dt>
                  <dd>Rank of the surface-energy candidate within this bulk, in the order {esc(surf_order)}
                    (1 = lowest &gamma;). Hover the pointer over a rank to see the model order.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">&gamma; checks</dt>
                  <dd>Surface-energy ranking agreement (<em>rank: same</em> / <em>differs</em> across models)
                    and the geometry-sensitivity status (<em>geometry: low</em> / <em>sensitive</em>, a heuristic
                    from the other models' residual forces on the {esc(surf_primary)} geometries). Hover the pointer over a
                    chip to see the per-model ranks or the geometry-sensitivity estimate behind it.</dd>
                </div>"""
    else:
        spread_bit = eta_spread_bit = unc_terms = ""
    per_bulk_terms = f"""
          <div class="rounded-xl border bg-white p-6">
            <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600">
                <div>
                  <dt class="font-semibold text-slate-900">Surface ID</dt>
                  <dd>Internal identifier for the generated slab or surface candidate.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Miller Index</dt>
                  <dd>Crystallographic orientation of the surface, expressed as (hkl).</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Surface Energy (eV/&Aring;&sup2;)</dt>
                  <dd>Energy per unit surface area, &gamma;, reported for {esc(surf_primary)}{spread_bit}.
                    Lower values indicate lower calculated surface energy under the stated model and slab
                    conventions.</dd>
                </div>{unc_terms}
                <div>
                  <dt class="font-semibold text-slate-900"># Atoms</dt>
                  <dd>Number of atoms in the clean slab on which the best candidate was computed: the stored
                    1&times;1 slab's atom count multiplied by n<sub>x</sub>&middot;n<sub>y</sub> of that
                    candidate's Repeat. It includes the supercell scaling and excludes the adsorbate atoms.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Area (&Aring;&sup2;)</dt>
                  <dd>Surface area of the same slab configuration, scaled in the same way
                    (1&times;1 area &times; n<sub>x</sub>&middot;n<sub>y</sub>).</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Best &eta;</dt>
                  <dd>Overpotential of the {esc(surf_primary)}-selected best (lowest-&eta;) candidate for this
                    surface{eta_spread_bit}.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Repeat (n<sub>x</sub>, n<sub>y</sub>)</dt>
                  <dd>In-plane repetition used to construct the clean slab on which the best adsorption
                    candidate was placed.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Adsorbate screen</dt>
                  <dd>What happened to the adsorption sites tried on this surface: sites where an adsorbate
                    could not be placed, relaxations that failed or did not converge, adsorption sets rejected
                    by the sanity checks (with adsorbate and reason), sets missing a pathway intermediate,
                    candidates above the &eta; limit, and the number of candidates stored. A rejected or failed
                    intermediate drops the whole adsorption site. <em>not screened</em>: the surface was not
                    among the lowest-energy surfaces of its bulk that are screened.</dd>
                </div>
            </dl>
          </div>"""

    n_light_cell = str(n_light) if electronic_present else "&mdash;"

    # -- gap models: names for the Pipeline Metadata "Light Screen" entry, the
    # model -> target-gap explanation for the Light Harvesting section ------
    def target_txt(target):
        return f"{esc(target)} gap" if target else "gap"

    light_screen_cell = esc(screen_models) if electronic_present else "not run"
    gap_models_note = ""
    if electronic_present and screen_targets:
        bits = []
        for k, gap_model, data, t in screen_targets:
            if k == "megnet_mfi":
                bits.append(f"{esc(k)}: {esc(gap_model)} model trained jointly on band gaps at four levels of "
                            "theory (PBE, GLLB-SC, HSE, SCAN; Chen et al. 2021); the level is chosen at prediction time"
                            + (f", here {esc(t)} (the hybrid HSE06 functional)" if t == "HSE" else
                               (f", here {esc(t)}" if t else "")))
            else:
                bits.append(f"{esc(k)}: {esc(gap_model)} trained on {esc(data) if data else 'its'} "
                            f"{target_txt(t)}s")
        gap_models_note = (
            ". ".join(bits) + ". "
            + ("The HSE label belongs to megnet_mfi only. " if screen_fidelity and len(screen_targets) > 1 else ""))

    # -- light-harvesting section (only when the optical screen ran) ---------
    if electronic_present:
        n_light_cols = 7 + (len(model_keys) if multi_model else 0)
        light_rows = []
        for s in summaries:
            e = s.get("electronic")
            uid = s["structure_uuid"]
            if not e:
                light_rows.append(
                    f'<tr class="border-t"><td class="px-4 py-3 font-mono text-xs">{esc(uid[:8])}</td>'
                    f'<td class="px-4 py-3 text-slate-400 italic" colspan="{n_light_cols - 1}">not screened</td></tr>')
                continue
            gap, gap_std = e.get("gap_eV"), e.get("gap_std_eV")
            gap_cell = "&mdash;" if gap is None else (
                f'{gap:.2f} &plusmn; {gap_std:.2f}' if gap_std else f'{gap:.2f}')
            per_model = e.get("gap_values_eV") or {}
            if gap is not None and per_model:
                gap_title = " · ".join(f"{k} {v:.2f}" for k, v in per_model.items()) + " eV"
                gap_cell = f'<span title="{esc(gap_title)}">{gap_cell}</span>'
            model_gap_cells = "".join(
                f'\n                        <td class="px-4 py-3">'
                f'{f"{per_model[k]:.2f}" if k in per_model else "&mdash;"}</td>'
                for k in (model_keys if multi_model else []))
            by_model = bulk_straddle_by_model(e, reaction, reaction_path)
            straddle_cell = straddle_badge(by_model)
            if by_model:
                required = next(iter(by_model.values()))["margin_required_V"]
                cb_cell = per_model_cell({k: v["u_red"] - v["margin_reduction_V"] for k, v in by_model.items()}, ".2f")
                vb_cell = per_model_cell({k: v["u_ox"] + v["margin_oxidation_V"] for k, v in by_model.items()}, ".2f")
                red_cell = per_model_cell({k: v["margin_reduction_V"] for k, v in by_model.items()}, "+.2f", required)
                ox_cell = per_model_cell({k: v["margin_oxidation_V"] for k, v in by_model.items()}, "+.2f", required)
            else:
                cb_cell = vb_cell = red_cell = ox_cell = "&mdash;"
            light_rows.append(f"""
                      <tr class="border-t">
                        <td class="px-4 py-3 font-mono text-xs" title="{esc(uid)}">{esc(uid[:8])}</td>
                        <td class="px-4 py-3">{gap_cell}</td>{model_gap_cells}
                        <td class="px-4 py-3">{straddle_cell}</td>
                        <td class="px-4 py-3">{cb_cell}</td>
                        <td class="px-4 py-3">{vb_cell}</td>
                        <td class="px-4 py-3">{red_cell}</td>
                        <td class="px-4 py-3">{ox_cell}</td>
                      </tr>""")

        couple_bit = esc(couple["label"]) if couple else "&mdash;"
        gap_models_txt = (", ".join(f"{esc(k)} &rarr; {target_txt(t)}" for k, _, _, t in screen_targets)
                          or esc(screen_models))
        margin_req = next((v["margin_required_V"] for s in summaries
                           for v in [bulk_straddle(s.get("electronic"), reaction, reaction_path)]
                           if v and v.get("margin_required_V") is not None), None)
        margin_txt = f"{margin_req:g} V" if margin_req is not None else "the required margin"
        names_txt = " / ".join(esc(model_names[k]) for k in model_keys)
        gap_mean_txt = (f"Plain mean of the ML-predicted band gaps ({gap_models_txt}), so it mixes their "
                        "target quantities. Shown for reference only: the straddle verdict, band edges and "
                        "margins are evaluated per model." if multi_model else
                        f"Mean of the ML-predicted band gaps ({gap_models_txt}).")

        if multi_model:
            partial_txt = "; ".join(
                f'<strong>{esc(uid[:8])}</strong> with {esc(", ".join(model_names.get(k, k) for k in passing))} only'
                for uid, passing in light_partial)
            straddle_count_txt = (f"Here <strong>{n_light_cell}</strong> of {len(summaries)} bulk(s) straddle with "
                                  f"every gap model, {len(light_partial)} with only some of them"
                                  + (f": {partial_txt}" if partial_txt else "") + ".")
        else:
            straddle_count_txt = f"Here <strong>{n_light_cell}</strong> of {len(summaries)} bulk(s) straddle."

        # chi is composition-only, so every bulk shares one mid-gap level and the
        # straddle test reduces to a single gap threshold for this composition
        mean_verdict = next((v for s in summaries
                             for v in [bulk_straddle(s.get("electronic"), reaction, reaction_path)] if v), None)
        mids = [0.5 * (ed["cb"] + ed["vb"]) for s in summaries
                for ed in [(s.get("electronic") or {}).get("band_edges_vs_rhe_V")] if ed]
        threshold_txt = ""
        if mean_verdict and mids and max(mids) - min(mids) < 0.01:
            mid, margin = mids[0], mean_verdict["margin_required_V"]
            need_red = 2 * (mid - mean_verdict["u_red"] + margin)
            need_ox = 2 * (mean_verdict["u_ox"] + margin - mid)
            limiting = ("conduction-band edge (reduction side)" if need_red >= need_ox
                        else "valence-band edge (oxidation side)")
            threshold_txt = (f"&chi; depends on composition only, so every bulk here has the same mid-gap level "
                             f"({mid:+.2f} V<sub>RHE</sub>) and straddling reduces to a gap threshold: "
                             f"E<sub>g</sub> &ge; <strong>{max(need_red, need_ox):.2f} eV</strong>, set by the "
                             f"{limiting}. ")
        if multi_model:
            targets = " vs ".join(esc(t) for _, _, _, t in screen_targets if t)
            threshold_txt += ("A model-dependent verdict mostly reflects the different levels of theory the gap "
                              f"models target{f' ({targets})' if targets else ''} and marks a candidate worth a "
                              "DFT check, not a result.")
        threshold_html = f"""
              <p>
                {threshold_txt}
              </p>""" if threshold_txt else ""

        model_sub = (f'<br><span class="normal-case font-normal">{names_txt}</span>' if multi_model else "")
        model_gap_heads = "".join(
            f'\n                    <th class="px-4 py-3" title="{esc(k)}">{esc(model_names[k])} '
            f'({f"{esc(t)}, " if t and t not in model_names[k] else ""}eV)</th>'
            for k, _, _, t in (screen_targets if multi_model else []))
        model_gap_note = f"""
                <div>
                  <dt class="font-semibold text-slate-900">{names_txt} (eV)</dt>
                  <dd>Each gap model's own band-gap prediction ({gap_models_txt}). The models target different
                    levels of theory, so part of their difference is systematic rather than random. Band edges,
                    Straddle and the margins are evaluated with each of these gaps separately.</dd>
                </div>""" if multi_model else ""
        gap_source_txt = (f"Gaps from pretrained ML models. {gap_models_note.strip()}" if gap_models_note
                          else f"Gaps from pretrained ML models ({gap_models_txt}).")
        light_notes = f"""
              <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600 mt-5">
                <div>
                  <dt class="font-semibold text-slate-900">Bulk</dt>
                  <dd>Bulk structure id (first 8 characters; hover the pointer over an id to see the full UUID).
                    <em>not screened</em>: the optical screen produced no result for this bulk.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">{"Mean gap" if multi_model else "Gap"} (eV)</dt>
                  <dd>{gap_mean_txt} &plusmn; is the standard deviation
                    over these models (for two models, half their difference): model disagreement, not a
                    calibrated confidence interval. Hover the pointer over a value to see the per-model values.</dd>
                </div>{model_gap_note}
                <div>
                  <dt class="font-semibold text-slate-900">Straddle</dt>
                  <dd>{(f"Evaluated separately with each model's gap. <em>straddles</em>: Red. margin and Ox. margin "
                        f"are at least {margin_txt} for every model. <em>model-dependent</em>: true for some models "
                        "only (named). <em>no</em>: at least one edge falls short for every model. Hover the pointer "
                        "over a verdict to see each model's margins.") if multi_model else
                       (f"<em>straddles</em>: both Red. margin and Ox. margin are at least {margin_txt}. "
                        "<em>no</em>: at least one edge falls short. Hover the pointer over a verdict to see both "
                        "margins.")}</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">E<sub>CB</sub> / E<sub>VB</sub> (V<sub>RHE</sub>)</dt>
                  <dd>Conduction- and valence-band edges estimated with the empirical Butler&ndash;Ginley / Mulliken
                    relation, as electrochemical potentials:
                    E<sub>CB</sub> = &chi; &minus; 4.5 eV &minus; E<sub>g</sub>/2 and E<sub>VB</sub> = E<sub>CB</sub> + E<sub>g</sub>,
                    with &chi; the geometric-mean Mulliken electronegativity of the composition and
                    E<sub>g</sub> {f"each model's gap ({names_txt})" if multi_model else "the Gap column"}. On this
                    scale a more negative value is a higher-lying (more reducing) level. &chi; depends on composition
                    only, so polymorphs differ only through their gap: every bulk shares the mid-gap level
                    &chi; &minus; 4.5 eV.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Red. margin (V)</dt>
                  <dd>U<sub>red</sub> &minus; E<sub>CB</sub>: how far the conduction-band edge sits above
                    (more negative than) the reduction level. Must be &ge; {margin_txt}; negative means the edge
                    is too low to drive the reduction half-reaction.{f" Given per model ({names_txt})." if multi_model else ""}
                    Values below {margin_txt} are in red.</dd>
                </div>
                <div>
                  <dt class="font-semibold text-slate-900">Ox. margin (V)</dt>
                  <dd>E<sub>VB</sub> &minus; U<sub>ox</sub>: how far the valence-band edge sits below
                    (more positive than) the oxidation level. Must be &ge; {margin_txt}; negative means the edge
                    is too high to drive the oxidation half-reaction.{f" Given per model ({names_txt})." if multi_model else ""}
                    Values below {margin_txt} are in red.</dd>
                </div>
              </dl>"""

        light_section = f"""
        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="sun" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Light Harvesting</h2>
          </div>
          <div class="bg-white border rounded-xl p-6">
            <div class="border-l-4 border-primary bg-blue-50 text-slate-700 text-sm rounded-r-lg p-4 mb-5 space-y-2">
              <p>
                A photocatalyst can drive this reaction under illumination only if its band
                gap <strong>straddles</strong> the redox window: the conduction-band edge
                (E<sub>CB</sub>) must sit above the reduction level and the valence-band edge
                (E<sub>VB</sub>) below the oxidation level, each by at least the required
                margin. Target for <strong>{esc(reaction)} / {esc(reaction_path)}</strong>:
                {couple_bit}.
                {straddle_count_txt}
              </p>{threshold_html}
              <p class="text-sm text-slate-500">
                {gap_source_txt}
              </p>
            </div>
            <div class="border-l-4 border-amber-500 bg-amber-50 text-amber-900 text-sm rounded-r-lg p-4 mb-3">
              <strong>Screening limitation:</strong> Band gaps are ML predictions, and band edges are
              estimated using an empirical relation rather than calculated directly for each structure
              using a consistent electronic-structure method. The band-edge estimates have an expected
              uncertainty of approximately 0.3&ndash;0.5 eV and should be used to rank candidates, not
              establish photocatalytic feasibility.
            </div>
            <div class="overflow-x-auto border rounded-lg">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 text-xs">
                  <tr>
                    <th class="px-4 py-3">Bulk</th>
                    <th class="px-4 py-3">{"Mean gap" if multi_model else "Gap"} (eV)</th>{model_gap_heads}
                    <th class="px-4 py-3">Straddle</th>
                    <th class="px-4 py-3">E<sub>CB</sub> (V<sub>RHE</sub>){model_sub}</th>
                    <th class="px-4 py-3">E<sub>VB</sub> (V<sub>RHE</sub>){model_sub}</th>
                    <th class="px-4 py-3">Red. margin (V){model_sub}</th>
                    <th class="px-4 py-3">Ox. margin (V){model_sub}</th>
                  </tr>
                </thead>
                <tbody>{"".join(light_rows)}</tbody>
              </table>
            </div>{light_notes}
          </div>
        </section>"""
    else:
        light_section = ""

    synth_section, synth_models = _synthesizability_section(summaries, esc)
    phonon_section, phonon_models = _phonon_section(summaries, esc)
    uncertainty_section = _uncertainty_section(summaries, esc)
    surface_adsorption_section = _surface_adsorption_section(summaries, esc)
    analysed_candidates = [c["uncertainty"] for s in summaries
                           for c in s["reaction_candidates"] if c.get("uncertainty")]
    if analysed_candidates:
        n_robust_cand = sum(1 for u in analysed_candidates if u.get("label") == "robust")
        robust_candidates_tile = f"""
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100" title="Reaction candidates with the same potential-determining step and std(eta) within tolerance under every committee MLIP">
                <p class="text-xs font-bold text-slate-500 mb-1">Robust candidates</p>
                <p class="text-lg font-bold text-slate-900">{n_robust_cand} / {len(analysed_candidates)}</p>
              </div>"""
    else:
        robust_candidates_tile = ""
    if unc_present:
        robust_tile = f"""
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100" title="Bulks stable under every committee MLIP">
                <p class="text-xs font-bold text-slate-500 mb-1">Robust bulks</p>
                <p class="text-lg font-bold text-slate-900">{n_robust} / {n_unc}</p>
              </div>"""
    else:
        robust_tile = ""
    summary_cols = f"md:grid-cols-{3 + bool(unc_present) + bool(analysed_candidates)}"
    # One definition per tile, in tile order.
    summary_terms = [
        ("Bulk candidates", "Relaxed bulk structures screened in this run."),
        ("Surfaces retained", "Surface slabs kept after relaxation, summed over all bulks. Slabs whose "
                              "relaxation did not converge are dropped."),
        ("Reaction candidates", "Adsorption configurations (facet, site, coverage) with a predicted "
                                "overpotential &eta; &le; 2.0 V."),
    ]
    if unc_present:
        summary_terms.append(("Robust bulks", "Bulks predicted stable under every committee MLIP "
                                              "(see <em>Stability Sensitivity</em>)."))
    if analysed_candidates:
        summary_terms.append(("Robust candidates",
                              "Reaction candidates with the same potential-determining step under every "
                              "committee MLIP and &sigma;<sub>&eta;</sub> within tolerance "
                              "(see <em>Surface &amp; Adsorption Uncertainty</em>)."))
    summary_note = "".join(f"""
              <div>
                <dt class="font-semibold text-slate-900">{dt}</dt>
                <dd>{dd}</dd>
              </div>""" for dt, dd in summary_terms)

    # "ZnO — CO2RR — CO reduction to CH₄"; an unlisted path is shown by its bare name
    # (pathway_title's "REACTION / path" fallback would repeat the reaction).
    path_title = (_PATHWAY_TITLES.get((reaction or "").strip().upper(), {})
                  .get((reaction_path or "").strip().lower()) or reaction_path)
    page_title = f"{esc(chemical_formula)} &mdash; {esc(reaction)} &mdash; {esc(path_title)}"

    html_doc = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{page_title}</title>

    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {{ theme: {{ extend: {{ colors: {{ primary: "#2563eb" }} }} }} }};
    </script>
    <script src="https://unpkg.com/lucide@latest"></script>
  </head>

  <body class="flex flex-col min-h-screen bg-slate-50">
    <div class="bg-white border-b px-6 py-6 sticky top-0 z-10">
      <div class="max-w-screen-2xl mx-auto flex flex-col gap-2">
        <h1 class="text-slate-900 text-2xl font-bold tracking-tight" title="{esc(reaction)} / {esc(reaction_path)}">
          {page_title}
        </h1>
      </div>
    </div>

    <div class="p-6 md:p-12 max-w-screen-2xl mx-auto w-full grid grid-cols-1 lg:grid-cols-4 gap-8">
      <div class="lg:col-span-3 flex flex-col gap-8">

        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="info" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Executive Summary</h2>
          </div>
          <div class="rounded-xl border bg-white shadow-sm p-6">
            <div class="grid grid-cols-2 {summary_cols} gap-4">
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-xs font-bold text-slate-500 mb-1">Bulk candidates</p>
                <p class="text-lg font-bold text-slate-900">{n_bulks}</p>
              </div>
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-xs font-bold text-slate-500 mb-1">Surfaces retained</p>
                <p class="text-lg font-bold text-slate-900">{n_surfaces}</p>
              </div>
              <div class="p-3 rounded-xl bg-slate-50 border border-slate-100">
                <p class="text-xs font-bold text-slate-500 mb-1">Reaction candidates</p>
                <p class="text-lg font-bold text-slate-900">{n_candidates}</p>
              </div>{robust_tile}{robust_candidates_tile}
            </div>{no_candidate_note}{hull_gap_note}
            <dl class="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-slate-600 mt-5 pt-5 border-t border-slate-100">{summary_note}
            </dl>
          </div>
        </section>

        <section>
          <div class="flex items-center gap-2 mb-4">
            <i data-lucide="layers" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Bulk Structure Candidates</h2>
          </div>
          <div class="bg-white border rounded-xl overflow-hidden">
            <div class="overflow-x-auto">
              <table class="w-full text-sm text-left text-slate-700">
                <thead class="bg-slate-50 text-slate-500 text-xs">
                  <tr>
                    <th class="px-6 py-3" title="Identifier of the bulk structure in the uvsib database; first 8 characters shown, hover a cell for the full UUID">UUID</th>
                    <th class="px-6 py-3" title="MPDB(exp.): experimentally observed, Materials Project · MPDB(stb.): stable, Materials Project · CSP: crystal structure prediction · generated: ML-generated">Source</th>
                    <th class="px-6 py-3" title="Materials Project id (MPDB-sourced bulks only)">MP ID</th>
                    <th class="px-6 py-3" title="± = standard deviation across the MLIP committee (model disagreement, not a DFT-calibrated interval)">E above hull (eV/atom)</th>
                    <th class="px-6 py-3" title="Surface slab models retained for this bulk; 0 does not mean the material has no stable surfaces">Surfaces retained</th>
                  </tr>
                </thead>
                <tbody>{"".join(bulk_rows) or f'<tr><td colspan="5" class="px-6 py-4 text-slate-400 italic">no bulk candidates found</td></tr>'}</tbody>
              </table>
            </div>
            <div class="border-t bg-slate-50 px-6 py-4">{bulk_notes}
            </div>
          </div>
        </section>

        {uncertainty_section}

        {phonon_section}

        {synth_section}

        {light_section}

        {surface_adsorption_section}

        <section class="flex flex-col gap-6">
          <div class="flex items-center gap-2">
            <i data-lucide="bar-chart-3" class="w-5 h-5 text-primary"></i>
            <h2 class="text-xl font-bold text-slate-900">Per-Bulk Detail: Surfaces &amp; Reaction Paths</h2>
          </div>{per_bulk_terms}
          {"".join(bulk_sections)}
        </section>

      </div>

      <div class="flex flex-col gap-8">
        <div class="rounded-xl border bg-white shadow-sm p-6">
          <h3 class="text-lg font-bold text-slate-900 mb-6">Metadata</h3>
          <div class="space-y-5">
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center shrink-0">
                <i data-lucide="beaker" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">Composition</p>
                <p class="text-sm font-bold text-slate-900">{esc(chemical_formula)}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center shrink-0">
                <i data-lucide="flask-conical" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">Reaction</p>
                <p class="text-sm font-bold text-slate-900">{esc(reaction)}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-orange-100 text-orange-600 flex items-center justify-center shrink-0">
                <i data-lucide="git-branch" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">Reaction Path</p>
                <p class="text-sm font-bold text-slate-900">{esc(reaction_path)}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center shrink-0">
                <i data-lucide="cpu" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">ML Bulk Model</p>
                <p class="text-sm font-bold text-slate-900">{esc(_model_label(model, bulk_task))}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center shrink-0">
                <i data-lucide="layers-3" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">ML Surface Model</p>
                <p class="text-sm font-bold text-slate-900">{esc(_model_label(surface_model, surface_task))}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center shrink-0">
                <i data-lucide="sun" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">Light Harvesting</p>
                <p class="text-sm font-bold text-slate-900">{light_screen_cell}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0">
                <i data-lucide="test-tube" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">Synthesizability</p>
                <p class="text-sm font-bold text-slate-900">{esc(synth_models) if synth_models else "not run"}</p>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="size-10 rounded-full bg-violet-100 text-violet-600 flex items-center justify-center shrink-0">
                <i data-lucide="activity" class="w-6 h-6"></i>
              </div>
              <div>
                <p class="text-xs font-bold text-slate-400 capitalize tracking-wider">Phonons</p>
                <p class="text-sm font-bold text-slate-900">{esc(phonon_models) if phonon_models else "not run"}</p>
              </div>
            </div>
          </div>
        </div>

        <div class="rounded-xl bg-primary text-white p-6">
          <h3 class="text-lg font-bold mb-2">Need the raw data?</h3>
          <p class="text-white/80 text-sm mb-6">
            Download all numbers and structures behind this page &mdash; bulk and
            surface structures, E<sub>hull</sub>, surface energies, and &eta; &mdash; as
            one JSON file.
          </p>
          <a
            href="{esc(raw_data_filename)}"
            download
            class="block w-full text-center bg-white text-primary rounded-md py-2 font-bold"
          >
            Download
          </a>
        </div>

        <div class="rounded-xl border bg-white shadow-sm p-6">
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center gap-2">
              <i data-lucide="quote" class="w-5 h-5 text-primary"></i>
              <h3 class="text-lg font-bold text-slate-900">
                Cite This Project
              </h3>
            </div>
          </div>

          <p class="text-xs text-slate-500 mb-3">
            If you use UvSiB in your research, please cite our project and
            open-source repository:
          </p>

          <!-- TODO: Update BibTeX citation in this box -->
          <div
            class="relative bg-slate-900 text-slate-200 rounded-lg p-4 font-mono text-xs overflow-x-auto mb-4 border border-slate-800"
          >
            <pre id="bibtex-content">
@software{{uvsib2026,
  author = {{Research Team, UvSiB CASUS-HZDR}},
  title = {{UvSiB: Predictive Chemistry and Molecular Simulation Framework}},
  year = {{2026}},
  url = {{https://github.com/casus/uvsib-framework}},
  note = {{Open-source research platform}}
}}</pre
            >
          </div>

          <button
            type="button"
            onclick="copyBibtex()"
            class="w-full inline-flex items-center justify-center gap-2 bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold text-sm py-2.5 px-4 rounded-lg transition-all cursor-pointer border border-slate-200"
          >
            <i data-lucide="copy" id="copy-icon" class="w-4 h-4"></i>
            <span id="copy-text">Copy BibTeX</span>
          </button>
        </div>
      </div>
    </div>

    <script>
      lucide.createIcons();

      function copyBibtex() {{
        var text = document.getElementById("bibtex-content").innerText;
        var label = document.getElementById("copy-text");
        function done() {{
          label.textContent = "Copied!";
          setTimeout(function () {{ label.textContent = "Copy BibTeX"; }}, 2000);
        }}
        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(text).then(done);
        }} else {{
          var ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
          done();
        }}
      }}
    </script>
  </body>
</html>
"""

    raw_data_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), raw_data_filename)
    with open(raw_data_path, "w") as f:
        json.dump(raw_data(chemical_formula, reaction, reaction_path, summaries=summaries),
                   f, indent=2, default=str)

    html_doc = _keep_unit_case(html_doc)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return html_doc

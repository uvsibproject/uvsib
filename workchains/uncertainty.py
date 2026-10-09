"""EhullUncertaintyWorkChain -- E_above_hull uncertainty of the primary MLIP.

A ``PhaseDiagramMLWorkChain`` branch (run right after the ML bulk selection,
gated by ``settings.UNCERTAINTY_ENABLED``). For the composition's selected bulks
it estimates how much their E_above_hull depends on the choice of MLIP:

1. select, on the PRIMARY model's phase diagram (``ml_bulk_model``), the
   structures that can shape the selected candidates' hull: the elemental
   references, every entry of every subsystem within ``ehull_window`` of the
   primary hull, and the selected candidates (``ml_selection``);
2. submit, per committee model, single-point jobs (no relaxation) on the
   primary-relaxed geometries -- each committee model runs on its own code /
   environment, so every model is a separate job (split into chunks of
   ``chunk_size``);
3. store each result as a DBStructureVersion of the same structure_uuid with
   ``method = <committee model>`` (``add_version_to_existing_structure``);
   the forces / stress of the model on that geometry go to ``attributes``;
4. build one phase diagram per model on the same structure set and compute the
   per-model E_hull, the signed hull distance and their spread, plus the
   "geometry disagreement" flag from the estimated relaxation energies
   (``ehull_uncertainty.committee_ehull``);
5. write the summary to ``DBComposition.stable_struct["ml_uncertainty"]`` and
   each committee model's E_hull to its DBStructureVersion row.

Committee single points are reused across runs (competing phases are shared by
compositions of the same chemical system): a row is reused only if it is a
single point on the current primary geometry with the same checkpoint and head;
a stale single point is recomputed and overwritten. A row of the same method
that is NOT one of these single points (e.g. a structure relaxed by that model
in another run) is never overwritten -- that structure is excluded instead.

Advisory only: selection is unchanged and the parent never fails on this branch.
Config: ``input.yaml`` ``uncertainty:`` block (see settings.py).
"""

from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory

from uvsib.db.tables import DBComposition, DBStructureVersion
from uvsib.db.session import get_session
from uvsib.db.utils import (
        add_version_to_existing_structure,
        query_by_columns,
        update_row,
        get_chemical_systems,
        SINGLEPOINT_CALC)
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workchains.ehull_uncertainty import committee_ehull
from uvsib.workflows import settings

EHULL_ML = settings.EHULL_ML

# energy match tolerance (eV) for "this single point was done on the current
# primary geometry"
_GEOMETRY_ENERGY_TOL = 1e-6

# format of the stored single-point attributes; rows of an older version are
# treated as stale and recomputed (v2 added force_sq_mean + stress_voigt,
# needed by the relaxation-energy geometry check)
SP_VERSION = 2


def _rows_by_method(chemical_formula, method):
    """``{uuid: row-dict}`` for every DBStructureVersion of ``method`` in all
    chemical (sub)systems of ``chemical_formula``, elemental references
    (MPDB_ref) included. Rows without an energy are skipped."""
    chemical_systems, _ = get_chemical_systems(chemical_formula)
    rows = {}
    with get_session() as session:
        results = (
            session.query(DBStructureVersion)
            .filter(DBStructureVersion.chemsys.in_(chemical_systems))
            .filter(DBStructureVersion.method == method)
            .all()
        )
        for row in results:
            if row.energy is None:
                continue
            rows[str(row.structure_uuid)] = {
                "source": row.source,
                "structure": row.structure,
                "energy": float(row.energy),
                "attributes": dict(row.attributes or {}),
            }
    return rows


def _entry(uuid, row):
    struct = Structure.from_dict(row["structure"])
    return ComputedStructureEntry(structure=struct, energy=row["energy"],
                                  data={"struct_uuid": uuid})


def _is_singlepoint(row):
    return (row["attributes"] or {}).get("calc") == SINGLEPOINT_CALC


def _is_current(row, primary_row, primary, member):
    """True if ``row`` is a single point of ``member`` on ``primary_row``'s
    geometry with the current checkpoint / head."""
    attrs = row["attributes"] or {}
    return (attrs.get("calc") == SINGLEPOINT_CALC
            and attrs.get("sp_version") == SP_VERSION
            and attrs.get("geometry_from") == primary
            and attrs.get("model_name") == member["model_name"]
            and attrs.get("head") == member["head"]
            and attrs.get("geometry_energy") is not None
            and abs(attrs["geometry_energy"] - primary_row["energy"]) < _GEOMETRY_ENERGY_TOL)


def _chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


class EhullUncertaintyWorkChain(WorkChain):
    """Committee (single-point) E_above_hull uncertainty for the ML bulk selection."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("ml_bulk_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.select_entries,
            cls.run_singlepoints,
            cls.store_versions,
            cls.compute_uncertainty,
            cls.store_summary,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The uncertainty analysis did not finish successfully")
        spec.exit_code(301, "ERROR_NO_STRUCTURES_FOUND", message="No ML bulk selection for the given formula")
        spec.exit_code(302, "ERROR_MISSING_REFERENCES", message="Missing primary-model elemental references")
        spec.exit_code(303, "ERROR_NO_COMMITTEE_RESULTS", message="No committee model produced usable energies")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.primary = self.inputs.ml_bulk_model.value
        self.ctx.committee = []
        for member in settings.UNCERTAINTY_COMMITTEE:
            model_name, _, _ = get_model_device(member["model"])
            self.ctx.committee.append({"model": member["model"], "head": member["head"],
                                       "model_name": model_name})
        self.report(f"Running EhullUncertaintyWorkChain for {self.ctx.chemical_formula}: "
                    f"primary={self.ctx.primary}, "
                    f"committee={[m['model'] for m in self.ctx.committee]}")

        rows = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})
        selection = ((rows[0].stable_struct or {}).get("ml_selection") or []) if rows else []
        self.ctx.targets = [str(s["uuid"]) for s in selection]
        if not self.ctx.targets:
            self.report(f"No ML bulk selection found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND

    def select_entries(self):
        """Pick the structure set S on the primary hull and, per committee
        model, the uuids that still need a single point."""
        primary = self.ctx.primary
        primary_rows = _rows_by_method(self.ctx.chemical_formula, primary)

        entries = {u: _entry(u, r) for u, r in primary_rows.items()}
        refs = {u: r for u, r in primary_rows.items() if r["source"] == "MPDB_ref"}
        ref_elements = {el.symbol for u in refs for el in entries[u].composition.elements}
        elements = {el.symbol for el in Composition(self.ctx.chemical_formula).elements}
        missing = sorted(elements - ref_elements)
        if missing:
            self.report(f"ERROR: no {primary} elemental reference (MPDB_ref) for {missing}; "
                        "a DFT fallback would mix methods -- skipping the uncertainty analysis.")
            return self.exit_codes.ERROR_MISSING_REFERENCES

        try:
            pd = PhaseDiagram(list(entries.values()))
        except ValueError as exc:
            self.report(f"ERROR: cannot build the {primary} phase diagram: {exc}")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        window = settings.UNCERTAINTY_EHULL_WINDOW
        selected = set(refs)
        selected.update(u for u, e in entries.items() if pd.get_e_above_hull(e) <= window)
        absent = [u for u in self.ctx.targets if u not in primary_rows]
        if absent:
            self.report(f"WARNING: selected uuid(s) {absent} have no {primary} version; "
                        "they cannot be analysed.")
        selected.update(u for u in self.ctx.targets if u in primary_rows)
        self.ctx.selected = sorted(selected)
        self.report(f"{len(self.ctx.selected)} structure(s) selected for re-evaluation "
                    f"({len(refs)} elemental reference(s), window={window} eV/atom, "
                    f"{len(entries)} {primary} entries in the chemical system).")

        self.ctx.todo = {}
        self.ctx.conflicts = {}
        for member in self.ctx.committee:
            model = member["model"]
            existing = _rows_by_method(self.ctx.chemical_formula, model)
            new, override, conflicts = [], [], []
            for u in self.ctx.selected:
                row = existing.get(u)
                if row is None:
                    new.append(u)
                elif _is_current(row, primary_rows[u], primary, member):
                    continue
                elif _is_singlepoint(row):
                    override.append(u)
                else:
                    conflicts.append(u)
            self.ctx.todo[model] = {"new": new, "override": override}
            self.ctx.conflicts[model] = conflicts
            n_reused = len(self.ctx.selected) - len(new) - len(override) - len(conflicts)
            self.report(f"{model}: {len(new)} new, {len(override)} stale (recompute), "
                        f"{n_reused} reused single point(s).")
            if conflicts:
                self.report(f"WARNING: {model}: {len(conflicts)} uuid(s) already have a non-single-point "
                            f"'{model}' version (e.g. relaxed by {model}); they are not overwritten and "
                            "are excluded from the analysis.")

    def run_singlepoints(self):
        """One single-point job (or several chunks) per committee model, each
        on that model's own code."""
        primary_rows = _rows_by_method(self.ctx.chemical_formula, self.ctx.primary)
        self.ctx.jobs = []
        for k, member in enumerate(self.ctx.committee):
            model = member["model"]
            uuids = self.ctx.todo[model]["new"] + self.ctx.todo[model]["override"]
            if not uuids:
                continue
            payload = [{"uuid": u, "structure": primary_rows[u]["structure"], "index": i}
                       for i, u in enumerate(uuids)]
            for i, chunk in enumerate(_chunk(payload, settings.UNCERTAINTY_CHUNK_SIZE)):
                key = f"sp_{k}_{i}"
                builder = self._construct_singlepoint_builder(member, chunk, i)
                self.to_context(**{key: self.submit(builder)})
                self.ctx.jobs.append([key, model])
        self.report(f"Submitted {len(self.ctx.jobs)} single-point job(s).")

    def store_versions(self):
        """Store every single point as a DBStructureVersion (method = model)."""
        primary_rows = _rows_by_method(self.ctx.chemical_formula, self.ctx.primary)
        for key, model in self.ctx.jobs:
            node = self.ctx[key]
            if not node.is_finished_ok:
                self.report(f"{model}: single-point job {key} failed (exit {node.exit_status}); "
                            "its structures are excluded.")
                continue
            member = next(m for m in self.ctx.committee if m["model"] == model)
            override = set(self.ctx.todo[model]["override"])
            output = node.outputs.output_dict.get_dict()
            for fail in output.get("failed", []):
                self.report(f"{model}: single point failed for {fail.get('uuid')} ({fail.get('reason')})")

            n_stored = 0
            for rec in output.get("results", []):
                uuid = rec.get("uuid")
                if uuid not in primary_rows:
                    self.report(f"{model}: result for unknown uuid {uuid!r} -- dropped")
                    continue
                primary_row = primary_rows[uuid]
                add_attributes = {
                    "source": primary_row["source"],
                    "energy": rec["energy"],
                    "ehull": None,
                    "attributes": {
                        "calc": SINGLEPOINT_CALC,
                        "sp_version": SP_VERSION,
                        "geometry_from": self.ctx.primary,
                        "geometry_energy": primary_row["energy"],
                        "model_name": member["model_name"],
                        "head": member["head"],
                        "max_force": rec.get("max_force"),
                        "force_sq_mean": rec.get("force_sq_mean"),
                        "stress_voigt": rec.get("stress_voigt"),
                        "max_stress": rec.get("max_stress"),
                        "pressure": rec.get("pressure"),
                        "calc_pk": node.pk,
                    },
                }
                if uuid in override:
                    add_attributes["structure"] = primary_row["structure"]
                stored = add_version_to_existing_structure(
                    uuid, primary_row["structure"], model, add_attributes,
                    on_conflict="override" if uuid in override else "error")
                if stored:
                    n_stored += 1
                else:
                    self.report(f"{model}: a '{model}' version of {uuid} appeared meanwhile; "
                                "not overwritten.")
            self.report(f"{model}: {n_stored} single point(s) stored from {key}.")

    def compute_uncertainty(self):
        """Per-model phase diagrams on a common structure set + spread."""
        primary = self.ctx.primary
        selected = set(self.ctx.selected)
        primary_rows = _rows_by_method(self.ctx.chemical_formula, primary)

        entries = {primary: {u: _entry(u, r) for u, r in primary_rows.items() if u in selected}}
        diagnostics = {}
        self.ctx.used_committee = []
        for member in self.ctx.committee:
            model = member["model"]
            rows = _rows_by_method(self.ctx.chemical_formula, model)
            usable = {u: r for u, r in rows.items()
                      if u in selected and u in primary_rows
                      and _is_current(r, primary_rows[u], primary, member)}
            if not usable:
                self.report(f"WARNING: {model} has no usable single points; model dropped.")
                continue
            entries[model] = {u: _entry(u, r) for u, r in usable.items()}
            diagnostics[model] = {u: {k: r["attributes"].get(k)
                                      for k in ("max_force", "force_sq_mean", "stress_voigt",
                                                "max_stress", "pressure")}
                                  for u, r in usable.items()}
            self.ctx.used_committee.append(member)

        if not self.ctx.used_committee:
            self.report("ERROR: no committee model produced usable energies.")
            return self.exit_codes.ERROR_NO_COMMITTEE_RESULTS

        try:
            result = committee_ehull(
                self.ctx.targets, entries, diagnostics, primary,
                ehull_threshold=EHULL_ML,
                force_constant=settings.UNCERTAINTY_FORCE_CONSTANT,
                shear_modulus=settings.UNCERTAINTY_SHEAR_MODULUS,
                relax_energy_floor=settings.UNCERTAINTY_RELAX_ENERGY_FLOOR)
        except ValueError as exc:
            self.report(f"ERROR: committee phase diagrams failed: {exc}")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        if result["dropped"]:
            self.report(f"WARNING: {len(result['dropped'])} structure(s) missing for at least one "
                        "model were excluded from every model's hull.")
        if result["missing_targets"]:
            self.report(f"WARNING: selected uuid(s) {result['missing_targets']} could not be analysed "
                        "(missing for at least one model).")
        if not result["per_uuid"]:
            self.report("ERROR: none of the selected structures could be analysed.")
            return self.exit_codes.ERROR_NO_COMMITTEE_RESULTS
        self.ctx.result = result

    def store_summary(self):
        """``DBComposition.stable_struct["ml_uncertainty"]`` + committee E_hull
        on each committee DBStructureVersion row of the selected bulks."""
        result = self.ctx.result
        summary = {
            "primary": self.ctx.primary,
            "committee": [dict(m) for m in self.ctx.used_committee],
            "calc": "singlepoint",
            "ehull_threshold": float(EHULL_ML),
            "ehull_window": float(settings.UNCERTAINTY_EHULL_WINDOW),
            "force_constant": float(settings.UNCERTAINTY_FORCE_CONSTANT),
            "shear_modulus": float(settings.UNCERTAINTY_SHEAR_MODULUS),
            "relax_energy_floor": float(settings.UNCERTAINTY_RELAX_ENERGY_FLOOR),
            "n_selected": len(self.ctx.selected),
            "n_common": result["n_common"],
            "dropped": result["dropped"],
            "conflicts": {m: list(u) for m, u in self.ctx.conflicts.items() if u},
            "missing_targets": result["missing_targets"],
            "per_uuid": result["per_uuid"],
        }

        rows = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})
        if not rows:
            self.report(f"ERROR: {self.ctx.chemical_formula} was not found in DBComposition")
            return self.exit_codes.ERROR_CALCULATION_FAILED
        stable_struct = dict(rows[0].stable_struct or {})
        stable_struct["ml_uncertainty"] = summary
        update_row(DBComposition, rows[0].uuid, {"stable_struct": stable_struct})

        primary_rows = _rows_by_method(self.ctx.chemical_formula, self.ctx.primary)
        for uuid, res in result["per_uuid"].items():
            for member in self.ctx.used_committee:
                model = member["model"]
                add_version_to_existing_structure(
                    uuid, primary_rows[uuid]["structure"], model,
                    {"ehull": res["ehull"][model]}, on_conflict="override")

    def final_report(self):
        """One line per selected bulk"""
        methods = self.ctx.result["methods"]
        for uuid, res in self.ctx.result["per_uuid"].items():
            per_model = " ".join(f"{m}={res['ehull'][m]:.3f}" for m in methods)
            geometry = ""
            if res["geometry_disagreement"]:
                via = res["geometry_via"]
                geometry = (f" (geometry shift {1000 * res['geometry_shift_max']:.1f} meV/atom > "
                            f"{1000 * res['geometry_limit']:.1f}; {res['geometry_source']}: "
                            f"{via['uuid'][:8]} under {via['model']})")
            self.report(f"{uuid[:8]}: ehull {per_model} | std(ed)={res['ed_std']:.3f} "
                        f"votes={res['stable_votes']}/{res['n_models']} -> {res['label']}{geometry}")
        self.report(f"EhullUncertaintyWorkChain for {self.ctx.chemical_formula} finished successfully")

    ################################################################################
    def _construct_singlepoint_builder(self, member, chunk, index):
        """Single-point builder on the committee model's own code"""
        model = member["model"]
        Workflow = WorkflowFactory(model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(chunk)
        builder.code = get_code(model)
        builder.local_label = Str(f"singlepoint {self.ctx.chemical_formula} #{index}")
        model_name, model_path, device = get_model_device(model)
        builder.job_info = Dict({
            "job_type": "singlepoint",
            "ML_model": model,
            "model_name": model_name,
            "model_path": model_path,
            "model_head": member["head"],
            "device": device,
        })
        return builder

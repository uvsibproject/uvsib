"""Surface-energy and adsorption / overpotential uncertainty work chains.

SurfaceUncertaintyWorkChain (advisory step of SurfaceBuilderWorkChain, gated by
``settings.SURFACE_UNCERTAINTY_ENABLED``)
    For every stored DBSurface row of the composition: committee single points
    on the primary-relaxed slab and on the bulk it was cut from; per model
    gamma_m = (E_slab - N eps_bulk) / 2A; spread, facet-ranking agreement and
    the force-based geometry check (``surface_uncertainty.committee_surfaces``).

AdsorptionUncertaintyWorkChain (advisory step of AdsorbatesWorkChain, gated by
``settings.ADSORPTION_UNCERTAINTY_ENABLED``)
    For every stored DBSurfaceMLAdsorbate row of (composition, reaction, path):
    committee single points on the relaxed clean slab, intermediates and gas
    references of its ``adsorb_set``; per model eta / dG through the SAME
    reaction function the pipeline uses; spread, PDS agreement, best-candidate
    agreement and the geometry check
    (``surface_uncertainty.committee_adsorption``).

Both: no relaxation; one job per committee model on its own code (split into
chunks of ``uncertainty.chunk_size``); the committee is
``uncertainty.surface.committee``. Per-model results are cached in the row's
``attributes["uncertainty"]["models"][model]`` and reused while the model,
checkpoint, head and single-point format are unchanged. Nothing in the pipeline
reads these results -- selection and storage are unchanged.
"""

import hashlib
from ase.formula import Formula
from ase.io import jsonio
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory

from uvsib.db.tables import DBComposition, DBSurface, DBSurfaceMLAdsorbate
from uvsib.db.utils import query_by_columns, query_structure, merge_row_attributes, update_row
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workchains.surface_uncertainty import committee_surfaces, committee_adsorption
from uvsib.workchains.adsorbates import REACTION_FUNCTIONS
from uvsib.workflows import settings

# format of the per-model single-point entries in the row attributes; entries
# of another version are recomputed
SP_VERSION = 1


def _committee():
    members = []
    for member in settings.SURFACE_UNCERTAINTY_COMMITTEE:
        model_name, _, _ = get_model_device(member["model"])
        members.append({"model": member["model"], "head": member["head"],
                        "model_name": model_name})
    return members


def _is_current(entry, member):
    return (bool(entry) and entry.get("sp_version") == SP_VERSION
            and entry.get("model_name") == member["model_name"]
            and entry.get("head") == member["head"])


def _model_entry(row, model):
    return (((row.attributes or {}).get("uncertainty") or {}).get("models") or {}).get(model)


def _slab_entry_current(row, member):
    """Cached committee entry of a slab is valid: same model / checkpoint / head
    / format, computed on the slab geometry currently stored (same primary
    slab energy)."""
    entry = _model_entry(row, member["model"])
    return (_is_current(entry, member)
            and entry.get("slab_energy_primary") == (row.slab or {}).get("energy"))


def _chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _singlepoint_builder(member, payload, label):
    """Single-point builder on the committee model's own code."""
    model = member["model"]
    Workflow = WorkflowFactory(model.lower())
    builder = Workflow.get_builder()
    builder.input_structures = List(payload)
    builder.code = get_code(model)
    builder.local_label = Str(label)
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


def _submit_singlepoints(workchain, payload_by_model, label):
    """Submit chunked single-point jobs per model; returns [[ctx_key, model], ...]."""
    jobs = []
    for k, member in enumerate(workchain.ctx.committee):
        payload = payload_by_model.get(member["model"]) or []
        for i, chunk in enumerate(_chunk(payload, settings.UNCERTAINTY_CHUNK_SIZE)):
            key = f"sp_{k}_{i}"
            builder = _singlepoint_builder(member, chunk, f"{label} #{i}")
            workchain.to_context(**{key: workchain.submit(builder)})
            jobs.append([key, member["model"]])
    return jobs


def _collect_results(workchain):
    """``{model: {record_key: result}}`` from the finished single-point jobs."""
    results = {}
    for key, model in workchain.ctx.jobs:
        node = workchain.ctx[key]
        if not node.is_finished_ok:
            workchain.report(f"{model}: single-point job {key} failed (exit {node.exit_status}); "
                             "its structures are excluded.")
            continue
        output = node.outputs.output_dict.get_dict()
        for fail in output.get("failed", []):
            workchain.report(f"{model}: single point failed for {fail.get('uuid')} ({fail.get('reason')})")
        for rec in output.get("results", []):
            results.setdefault(model, {})[rec["uuid"]] = rec
    return results


def _slab_area(slab):
    a, b = slab["lattice"]["matrix"][0], slab["lattice"]["matrix"][1]
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return (cx ** 2 + cy ** 2 + cz ** 2) ** 0.5


def _bulk_method():
    return "r2SCAN" if settings._PD_VERIFICATION else settings.inputs["bulk_relax"]["model"]


def _merge_composition(chemical_formula, key, value):
    rows = query_by_columns(DBComposition, {"composition": chemical_formula})
    if not rows:
        return False
    stable_struct = dict(rows[0].stable_struct or {})
    stable_struct[key] = value
    update_row(DBComposition, rows[0].uuid, {"stable_struct": stable_struct})
    return True


class SurfaceUncertaintyWorkChain(WorkChain):
    """Committee (single-point) surface-energy uncertainty for the stored slabs."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_singlepoints,
            cls.store_singlepoints,
            cls.compute_and_store,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The surface uncertainty analysis failed")
        spec.exit_code(301, "ERROR_NO_SURFACES", message="No stored surfaces for the given formula")
        spec.exit_code(303, "ERROR_NO_COMMITTEE_RESULTS", message="No committee model produced usable energies")

    def setup(self):
        """Setup, report and work out which slabs each model still needs"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.primary = settings.inputs["face_build"]["model"]
        self.ctx.committee = _committee()
        rows = query_by_columns(DBSurface, {"composition": self.ctx.chemical_formula})
        self.report(f"Running SurfaceUncertaintyWorkChain for {self.ctx.chemical_formula}: "
                    f"{len(rows)} slab(s), primary={self.ctx.primary}, "
                    f"committee={[m['model'] for m in self.ctx.committee]}")
        if not rows:
            return self.exit_codes.ERROR_NO_SURFACES

        self.ctx.todo = {}
        for member in self.ctx.committee:
            todo = [row.id for row in rows if not _slab_entry_current(row, member)]
            self.ctx.todo[member["model"]] = todo
            self.report(f"{member['model']}: {len(todo)} slab(s) to evaluate, "
                        f"{len(rows) - len(todo)} reused.")

    def run_singlepoints(self):
        """Bulk + slab single points, one job (or chunks) per committee model"""
        rows = {row.id: row for row in query_by_columns(DBSurface, {"composition": self.ctx.chemical_formula})}
        bulk_method = _bulk_method()
        payload_by_model = {}
        for model, ids in self.ctx.todo.items():
            payload, bulks = [], set()
            for row_id in ids:
                row = rows.get(row_id)
                if row is None:
                    continue
                bulk_uuid = str(row.structure_uuid)
                if bulk_uuid not in bulks:
                    versions = query_structure({"uuid": bulk_uuid}, method=bulk_method)
                    if not versions:
                        self.report(f"No {bulk_method} bulk for {bulk_uuid}; its slabs are skipped.")
                        continue
                    bulks.add(bulk_uuid)
                    payload.append({"uuid": f"bulk:{bulk_uuid}", "structure": versions[0].structure,
                                    "index": len(payload)})
                payload.append({"uuid": f"slab:{row_id}", "structure": row.slab, "index": len(payload)})
            payload_by_model[model] = payload
        self.ctx.jobs = _submit_singlepoints(self, payload_by_model,
                                             f"surface singlepoint {self.ctx.chemical_formula}")
        self.report(f"Submitted {len(self.ctx.jobs)} single-point job(s).")

    def store_singlepoints(self):
        """Per model: gamma_m and force measures into the slab's attributes"""
        results = _collect_results(self)
        for member in self.ctx.committee:
            # re-read per model: each model's write must start from the
            # attributes the previous model just stored
            rows = {row.id: row for row in query_by_columns(DBSurface, {"composition": self.ctx.chemical_formula})}
            model = member["model"]
            res = results.get(model, {})
            n_stored = 0
            for row_id in self.ctx.todo[model]:
                row = rows.get(row_id)
                slab_rec = res.get(f"slab:{row_id}")
                bulk_rec = res.get(f"bulk:{row.structure_uuid}") if row else None
                if not slab_rec or not bulk_rec:
                    continue
                area = _slab_area(row.slab)
                n_atoms = slab_rec["n_atoms"]
                gamma = (slab_rec["energy"] - n_atoms * bulk_rec["epa"]) / (2.0 * area)
                unc = dict((row.attributes or {}).get("uncertainty") or {})
                models = dict(unc.get("models") or {})
                models[model] = {
                    "sp_version": SP_VERSION,
                    "model_name": member["model_name"],
                    "head": member["head"],
                    "slab_energy_primary": (row.slab or {}).get("energy"),
                    "slab_energy": slab_rec["energy"],
                    "epa_bulk": bulk_rec["epa"],
                    "gamma": gamma,
                    "slab_force_sq_sum": slab_rec.get("force_sq_sum"),
                    "slab_max_force": slab_rec.get("max_force"),
                    "bulk_force_sq_mean": bulk_rec.get("force_sq_mean"),
                    "bulk_max_force": bulk_rec.get("max_force"),
                }
                unc["models"] = models
                merge_row_attributes(DBSurface, row_id, {"uncertainty": unc})
                n_stored += 1
            self.report(f"{model}: surface energies stored for {n_stored} slab(s).")

    def compute_and_store(self):
        """Spread, ranking agreement and geometry check; write the summaries"""
        rows = query_by_columns(DBSurface, {"composition": self.ctx.chemical_formula})
        primary = self.ctx.primary
        slabs = []
        for row in rows:
            if row.formation_energy is None:
                continue
            gamma = {primary: float(row.formation_energy)}
            diag = {}
            for member in self.ctx.committee:
                if not _slab_entry_current(row, member):
                    continue
                entry = _model_entry(row, member["model"])
                gamma[member["model"]] = entry["gamma"]
                diag[member["model"]] = {k: entry.get(k) for k in
                                         ("slab_force_sq_sum", "bulk_force_sq_mean",
                                          "slab_max_force", "bulk_max_force")}
            if len(gamma) < 2:
                continue
            slabs.append({"surface_id": row.id, "bulk_uuid": str(row.structure_uuid),
                          "area": _slab_area(row.slab), "n_atoms": len(row.slab["sites"]),
                          "gamma": gamma, "diag": diag})
        if not slabs:
            self.report("ERROR: no slab has committee surface energies.")
            return self.exit_codes.ERROR_NO_COMMITTEE_RESULTS

        result = committee_surfaces(slabs, primary,
                                    force_constant=settings.UNCERTAINTY_FORCE_CONSTANT,
                                    gamma_floor=settings.SURFACE_UNCERTAINTY_GAMMA_FLOOR,
                                    n_selected=settings.MAX_NUM_ADS)
        attrs_by_id = {row.id: (row.attributes or {}) for row in rows}
        for surface_id, stats in result["per_slab"].items():
            unc = dict(attrs_by_id[surface_id].get("uncertainty") or {})
            unc.update(stats)
            merge_row_attributes(DBSurface, surface_id, {"uncertainty": unc})
        _merge_composition(self.ctx.chemical_formula, "surface_uncertainty", {
            "primary": primary,
            "head": settings.inputs["face_build"].get("head"),
            "committee": [dict(m) for m in self.ctx.committee],
            "calc": "singlepoint",
            "force_constant": float(settings.UNCERTAINTY_FORCE_CONSTANT),
            "gamma_floor": float(settings.SURFACE_UNCERTAINTY_GAMMA_FLOOR),
            "n_selected": int(settings.MAX_NUM_ADS),
            "per_bulk": result["per_bulk"],
        })
        self.ctx.result = result

    def final_report(self):
        """One line per slab"""
        for surface_id, res in self.ctx.result["per_slab"].items():
            per_model = " ".join(f"{m}={v:.4f}" for m, v in res["gamma"].items())
            geometry = ""
            if res["geometry_disagreement"]:
                geometry = (f" (geometry shift {res['geometry_shift_max']:.4f} > {res['geometry_limit']:.4f} "
                            f"eV/A^2; {res['geometry_source']} under {res['geometry_shift_model']})")
            self.report(f"slab {surface_id}: gamma {per_model} | std={res['gamma_std']:.4f} "
                        f"rank={res['rank']} -> {res['label']}{geometry}")
        for bulk_uuid, res in self.ctx.result["per_bulk"].items():
            self.report(f"bulk {bulk_uuid[:8]}: lowest-gamma facet {'agrees' if res['top1_agree'] else 'DIFFERS'} "
                        f"({res['top1']}), adsorption selection "
                        f"{'agrees' if res['selected_agree'] else 'DIFFERS'}")
        self.report(f"SurfaceUncertaintyWorkChain for {self.ctx.chemical_formula} finished successfully")


def _gas_molecule_count(atoms, name):
    """Molecules per gas-reference cell -- mirrors ``_reference_molecule_count``
    in codes/files/adsorbates.py (that module loads its reference files at
    import and cannot be imported here)."""
    return max(1, round(len(atoms) / len(Formula(name))))


def _decode_set(adsorb_set):
    """``[(species, json, atoms)]`` of one stored adsorb_set."""
    out = []
    for enc in adsorb_set.get("structures", []):
        atoms = jsonio.decode(enc)
        out.append((atoms.info["adsorbate"], enc, atoms))
    return out


def _structure_key(enc):
    return hashlib.sha1(enc.encode("utf-8")).hexdigest()[:20]


class AdsorptionUncertaintyWorkChain(WorkChain):
    """Committee (single-point) eta uncertainty for the stored reaction candidates."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_singlepoints,
            cls.store_singlepoints,
            cls.compute_and_store,
            cls.final_report,
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The adsorption uncertainty analysis failed")
        spec.exit_code(301, "ERROR_NO_CANDIDATES", message="No stored reaction candidates")
        spec.exit_code(302, "ERROR_UNKNOWN_REACTION", message="Unknown reaction")
        spec.exit_code(303, "ERROR_NO_COMMITTEE_RESULTS", message="No committee model produced usable energies")

    def _rows(self):
        return query_by_columns(DBSurfaceMLAdsorbate, {
            "composition": self.ctx.chemical_formula,
            "reaction": self.ctx.reaction,
            "reaction_path": self.ctx.reaction_path,
        })

    def setup(self):
        """Setup, report and work out which candidates each model still needs"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.reaction_path = self.inputs.reaction_path.value
        self.ctx.primary = settings.inputs["adsorbates"]["model"]
        self.ctx.committee = _committee()
        if self.ctx.reaction not in REACTION_FUNCTIONS:
            return self.exit_codes.ERROR_UNKNOWN_REACTION
        rows = self._rows()
        self.report(f"Running AdsorptionUncertaintyWorkChain for {self.ctx.chemical_formula} "
                    f"{self.ctx.reaction}/{self.ctx.reaction_path}: {len(rows)} candidate(s), "
                    f"primary={self.ctx.primary}, committee={[m['model'] for m in self.ctx.committee]}")
        if not rows:
            return self.exit_codes.ERROR_NO_CANDIDATES

        self.ctx.todo = {}
        for member in self.ctx.committee:
            todo = [row.id for row in rows
                    if not _is_current(_model_entry(row, member["model"]), member)]
            self.ctx.todo[member["model"]] = todo
            self.report(f"{member['model']}: {len(todo)} candidate(s) to evaluate, "
                        f"{len(rows) - len(todo)} reused.")

    def run_singlepoints(self):
        """Unique structures of the candidates, one job (or chunks) per model"""
        rows = {row.id: row for row in self._rows()}
        payload_by_model = {}
        for model, ids in self.ctx.todo.items():
            payload, seen = [], set()
            for row_id in ids:
                structures = _decode_set(rows[row_id].adsorb_set)
                n_clean = next((len(a) for name, _, a in structures if name == "*"), None)
                for name, enc, atoms in structures:
                    key = _structure_key(enc)
                    if key in seen:
                        continue
                    seen.add(key)
                    n_ads = (len(atoms) - n_clean
                             if name.startswith("*") and name != "*" and n_clean else None)
                    payload.append({"uuid": key, "atoms": enc, "n_ads": n_ads, "index": len(payload)})
            payload_by_model[model] = payload
        self.ctx.jobs = _submit_singlepoints(
            self, payload_by_model,
            f"adsorbate singlepoint {self.ctx.chemical_formula} {self.ctx.reaction_path}")
        self.report(f"Submitted {len(self.ctx.jobs)} single-point job(s).")

    def store_singlepoints(self):
        """Per model and candidate: species energies and force measures"""
        results = _collect_results(self)
        for member in self.ctx.committee:
            # re-read per model: each model's write must start from the
            # attributes the previous model just stored
            rows = {row.id: row for row in self._rows()}
            model = member["model"]
            res = results.get(model, {})
            n_stored = 0
            for row_id in self.ctx.todo[model]:
                row = rows[row_id]
                energies, force_sq, max_ads = {}, {}, {}
                complete = True
                for name, enc, atoms in _decode_set(row.adsorb_set):
                    rec = res.get(_structure_key(enc))
                    if rec is None:
                        complete = False
                        break
                    n_mol = 1 if name.startswith("*") else _gas_molecule_count(atoms, name)
                    energies[name] = rec["energy"] / n_mol
                    force_sq[name] = (rec.get("force_sq_sum") or 0.0) / n_mol
                    if rec.get("max_force_ads") is not None:
                        max_ads[name] = rec["max_force_ads"]
                if not complete:
                    self.report(f"{model}: candidate {row_id} incomplete (a single point failed); skipped.")
                    continue
                unc = dict((row.attributes or {}).get("uncertainty") or {})
                models = dict(unc.get("models") or {})
                models[model] = {
                    "sp_version": SP_VERSION,
                    "model_name": member["model_name"],
                    "head": member["head"],
                    "energies": energies,
                    "force_sq_sum": force_sq,
                    "max_force_ads": max_ads,
                }
                unc["models"] = models
                merge_row_attributes(DBSurfaceMLAdsorbate, row_id, {"uncertainty": unc})
                n_stored += 1
            self.report(f"{model}: energies stored for {n_stored} candidate(s).")

    def compute_and_store(self):
        """eta spread, PDS / best-candidate agreement, geometry check"""
        calc_fn = REACTION_FUNCTIONS[self.ctx.reaction][0]
        primary = self.ctx.primary
        primary_key = f"{primary.lower()}_energy"
        rows = self._rows()
        candidates = []
        for row in rows:
            energies = {primary: {name: atoms.info[primary_key]
                                  for name, _, atoms in _decode_set(row.adsorb_set)}}
            force_sq, max_ads = {}, {}
            for member in self.ctx.committee:
                entry = _model_entry(row, member["model"])
                if not _is_current(entry, member):
                    continue
                energies[member["model"]] = entry["energies"]
                force_sq[member["model"]] = entry.get("force_sq_sum") or {}
                max_ads[member["model"]] = entry.get("max_force_ads") or {}
            if len(energies) < 2:
                continue
            candidates.append({"row_id": str(row.id), "bulk_uuid": str(row.structure_uuid),
                               "surface_id": row.surface_id, "energies": energies,
                               "force_sq_sum": force_sq, "max_force_ads": max_ads})
        if not candidates:
            self.report("ERROR: no candidate has committee energies.")
            return self.exit_codes.ERROR_NO_COMMITTEE_RESULTS

        result = committee_adsorption(
            candidates, primary, calc_fn, self.ctx.reaction_path,
            force_constant=settings.UNCERTAINTY_FORCE_CONSTANT,
            eta_tolerance=settings.ADSORPTION_UNCERTAINTY_ETA_TOLERANCE,
            eta_floor=settings.ADSORPTION_UNCERTAINTY_ETA_FLOOR)
        for fail in result["failed"]:
            self.report(f"candidate {fail['row_id']} under {fail['model']}: {fail['reason']}")

        # results stay per row (with the run settings in "meta"): reaction paths
        # can run in parallel, so a shared DBComposition summary would race; the
        # per-bulk view is derived from the rows (adsorption_bulk_summary)
        meta = {
            "primary": primary,
            "head": settings.inputs["adsorbates"].get("head"),
            "committee": [dict(m) for m in self.ctx.committee],
            "calc": "singlepoint",
            "force_constant": float(settings.UNCERTAINTY_FORCE_CONSTANT),
            "eta_tolerance": float(settings.ADSORPTION_UNCERTAINTY_ETA_TOLERANCE),
            "eta_floor": float(settings.ADSORPTION_UNCERTAINTY_ETA_FLOOR),
        }
        attrs_by_id = {str(row.id): (row.attributes or {}) for row in rows}
        for row_id, stats in result["per_candidate"].items():
            unc = dict(attrs_by_id[row_id].get("uncertainty") or {})
            unc.update(stats)
            unc["meta"] = meta
            merge_row_attributes(DBSurfaceMLAdsorbate, int(row_id), {"uncertainty": unc})
        self.ctx.result = result

    def final_report(self):
        """One line per candidate"""
        for row_id, res in self.ctx.result["per_candidate"].items():
            per_model = " ".join(f"{m}={v:.2f}" for m, v in res["eta"].items())
            geometry = ""
            if res["geometry_disagreement"]:
                via = res["geometry_via"]
                geometry = (f" (geometry shift {res['geometry_shift_max']:.2f} V > {res['geometry_limit']:.2f}; "
                            f"{res['geometry_source']} {via['species']} under {via['model']})")
            self.report(f"candidate {row_id}: eta {per_model} | std={res['eta_std']:.2f} "
                        f"PDS {'agree' if res['pds_agree'] else 'differ'} -> {res['label']}{geometry}")
        for bulk_uuid, res in self.ctx.result["per_bulk"].items():
            self.report(f"bulk {bulk_uuid[:8]}: best candidate {'agrees' if res['best_agree'] else 'DIFFERS'} "
                        f"({res['best']})")
        self.report(f"AdsorptionUncertaintyWorkChain for {self.ctx.chemical_formula} finished successfully")

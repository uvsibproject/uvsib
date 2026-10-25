from pymatgen.core import Composition
from aiida.orm import Str, List, Dict
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from uvsib.db.utils import (
        add_version_to_existing_structure,
        query_structure)
from uvsib.workchains.utils import (
        get_output_as_entry,
        get_ref_entries,
        get_code,
        add_from_mpdb,
        competing_chemical_systems,
        get_model_device)
from uvsib.workflows import settings

_MPDB_SOURCES = ("MPDB_stb", "MPDB_exp")

def _relaxed_uuids(structure_filters, ml_bulk_model):
    """structure_uuids that already have a ``ml_bulk_model`` version."""
    return {r.structure_uuid for r in query_structure(structure_filters, method=ml_bulk_model)}

def get_struct_uuid(chemical_formula, ml_bulk_model=None):
    """MPDB structures of the target formula still to be ML-relaxed."""
    done = _relaxed_uuids({"composition": chemical_formula}, ml_bulk_model) if ml_bulk_model else set()
    struct_uuid = []
    results = query_structure({"composition": chemical_formula}, method = "DFT")
    for r in results:
        if r.source in _MPDB_SOURCES and r.structure_uuid not in done:
            struct_uuid.append((r.structure, r.source, r.structure_uuid))
    return struct_uuid

def get_competing_struct_uuid(chemical_formula, ml_bulk_model):
    """MPDB structures of the other compositions in the target's subsystems
    (competing phases) still to be ML-relaxed."""
    struct_uuid = []
    for chemsys in competing_chemical_systems(chemical_formula):
        done = _relaxed_uuids({"chemsys": chemsys}, ml_bulk_model)
        for r in query_structure({"chemsys": chemsys}, method="DFT"):
            if (r.source in _MPDB_SOURCES and r.composition != chemical_formula
                    and r.structure_uuid not in done):
                struct_uuid.append((r.structure, r.source, r.structure_uuid))
    return struct_uuid

def get_ref_struct_uuid(chemical_formula, ml_bulk_model):
    struct_uuid = []
    missing_refs = []

    elements = [element.symbol for element in Composition(chemical_formula).elements]
    for el in elements:
        ML_result = query_structure({"chemsys": el}, source="MPDB_ref", method=ml_bulk_model)
        if ML_result:
            continue

        mpdb_result = query_structure({"chemsys": el}, source="MPDB_ref", method="DFT")
        if not mpdb_result:
            missing_refs.append(el)
            continue

        r = mpdb_result[0]
        struct_uuid.append((r.structure, r.source, r.structure_uuid))

    return struct_uuid, missing_refs

class MPDBMLWorkChain(WorkChain):
    """Work chain for ML relaxation of MPDB structures"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("ml_bulk_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_relax_mpdb_structures,
            cls.store_ml_energies,
            cls.final_report
        )

        spec.exit_code(300, "ERROR_CALCULATION_FAILED", message="The WorkChain did not finish successfully")
        spec.exit_code(302, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")
        spec.exit_code(303, "ERROR_MISSING_REFERENCE_STRUCTURES", message="Missing elemental reference structures")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.ml_bulk_model = self.inputs.ml_bulk_model.value
        self.report(f"Running ML relaxation WorkChain for MPDB structures for {self.ctx.chemical_formula}")
        n_competing, error = add_from_mpdb(self.ctx.chemical_formula)
        if error:
            self.report(f"WARNING: MPDB competing-phase query failed ({error}); "
                        "continuing with the generated competing phases only.")
        elif settings.MPDB_CHEMSYS_ENABLED:
            self.report(f"{n_competing} new MPDB competing-phase structures added.")
        self.ctx.struct_uuid = get_struct_uuid(self.ctx.chemical_formula, self.ctx.ml_bulk_model)
        if not self.ctx.struct_uuid:
            self.report(
                    f"No MPDB structures of {self.ctx.chemical_formula} left to relax."
            )
        else:
            self.report(f"{len(self.ctx.struct_uuid)} structures from the MPDB was found for {self.ctx.chemical_formula}.")
        ref_structs, missing_refs = get_ref_struct_uuid(self.ctx.chemical_formula, self.ctx.ml_bulk_model)
        if missing_refs:
            self.report(f"ERROR: missing elemental reference structures for {missing_refs}.")
            return self.exit_codes.ERROR_MISSING_REFERENCE_STRUCTURES
        if ref_structs:
            self.ctx.struct_uuid.extend(ref_structs)
            self.report(f"{len(ref_structs)} reference structures")
        # competing phases go last; unlike the target and reference structures,
        # one that does not converge is dropped instead of failing the run
        self.ctx.n_required = len(self.ctx.struct_uuid)
        if settings.MPDB_CHEMSYS_ENABLED:
            competing = get_competing_struct_uuid(self.ctx.chemical_formula, self.ctx.ml_bulk_model)
            if competing:
                self.report(f"{len(competing)} MPDB competing-phase structures to relax")
            self.ctx.struct_uuid.extend(competing)

    def run_relax_mpdb_structures(self):
        """Optimize structures from MPDB"""
        if not self.ctx.struct_uuid:
            return

        structs = []
        for s, _, _ in self.ctx.struct_uuid:
            structs.append(s)
        builder = self._construct_ML_relax_builder(structs)
        self.to_context(**{"ml_e": self.submit(builder)})

    def store_ml_energies(self):
        """Collect ML-calculated energies"""
        if "ml_e" not in self.ctx:
            return

        ml_bulk_model = self.ctx.ml_bulk_model
        wch = self.ctx.ml_e
        if not wch.is_finished_ok:
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            new_entries = get_output_as_entry(wch)
            indices = wch.outputs.output_dict.get_dict().get("indices")
        except Exception:
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        if not indices or len(indices) != len(new_entries):
            # legacy relax output: positional, every structure must converge
            if len(new_entries) != len(self.ctx.struct_uuid):
                self.report("ERROR: ML relaxation for some structures failed")
                return self.exit_codes.ERROR_ML_RELAX_FAILED
            indices = range(len(new_entries))

        converged = dict(zip(indices, new_entries))
        # workchains started before competing phases existed: all required
        n_required = self.ctx.get("n_required", len(self.ctx.struct_uuid))
        missing = [i for i in range(len(self.ctx.struct_uuid)) if i not in converged]
        if any(i < n_required for i in missing):
            self.report("ERROR: ML relaxation for some target/reference structures failed")
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        if missing:
            self.report(f"WARNING: {len(missing)} MPDB competing-phase structures did not converge; "
                        "they are left out of the hull.")

        failed_stores = []
        for i, entry in converged.items():
            _, source, struct_uuid = self.ctx.struct_uuid[i]
            on_conflict = "error" if i < n_required and source != "MPDB_ref" else "ignore"
            stored = add_version_to_existing_structure(
                    struct_uuid,
                    entry.structure.as_dict(),
                    ml_bulk_model,
                    {
                     "source": source,
                     "energy": entry.energy,
                     },
                    on_conflict=on_conflict
            )

            if not stored:
                failed_stores.append(struct_uuid)

        if failed_stores:
            self.report(
              f"ERROR: failed to store {len(failed_stores)} "
              f"{ml_bulk_model} MPDB structure versions."
            )
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def final_report(self):
        """Final report"""
        _, missing = get_ref_entries(self.ctx.chemical_formula, self.ctx.ml_bulk_model)
        if missing:
            self.report(f"Warning: DFT-fallback elemental refs for {missing} (per-element offset risk)")
        self.report("ML relaxation WorkChain for MPDB structures finished successfully")

    ################################################################################
    def _construct_ML_relax_builder(self, structures):
        """
        General builder for structure optimization with an ML model
        """
        ML_model = self.ctx.ml_bulk_model
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)
        builder.local_label = Str(f"relax {self.ctx.chemical_formula}")
        model, model_path, device = get_model_device(ML_model)

        job_info = {
            "job_type": "relax",
            "ML_model": ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["bulk_relax"]["head"],
            "device": device,
            "fmax": settings.inputs["bulk_relax"]["fmax"],
            "max_steps": settings.inputs["bulk_relax"]["max_steps"]
        }

        builder.job_info = Dict(job_info)
        return builder

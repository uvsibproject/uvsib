from aiida.orm import Int, Str, List, Dict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain
from uvsib.db.utils import update_row, add_structures, query_by_columns
from uvsib.db.tables import DBChemsys
#from uvsib.codes.utils import get_mp_element_structures
from uvsib.workchains.utils import (unique_low_energy_chemsys,
                                    get_output_as_entry,
                                    get_ref_entries,
                                    get_code,
                                    get_model_device)
from uvsib.workflows import settings

EHULL_ML = settings.EHULL_ML

class GeneratorWorkChain(WorkChain):
    """Work chain for generating structures"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("ml_bulk_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.generative_calcs,
            cls.inspect_gen_calcs,
            cls.predict_ml_energies,
            cls.store_ml_energies,
            cls.final_report
        )

        spec.exit_code(300, "ERROR_GENERATIVE_FAILED", message="Generative calculation failed (no generator produced structures)")
        spec.exit_code(301, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")
        spec.exit_code(302, "ERROR_NO_CHEMSYS_FOUND", message="Chemical system does not exist in DBChemsys")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.ml_bulk_model = self.inputs.ml_bulk_model.value
        self.ctx.failed_ml_e = []
        self.report(f"Launching MatterGenWorkChain for {self.ctx.chemical_systems}")
        self.ctx.ref_entries, _ = get_ref_entries(self.ctx.chemical_formula, self.ctx.ml_bulk_model)

    def generative_calcs(self):
        """Run the generative workchains for each unique chemical system.

        MatterGen and GNoME (SAPS) are independent generators, each toggled in
        input.yaml (`mattergen.gen_enabled`, `gnome.enabled`). Whichever are
        enabled run in parallel for the same system and their candidates are
        merged before the shared ML relaxation. At least one must be enabled.
        """
        if not (settings.MATTERGEN_GEN_ENABLED or settings.GNOME_PARALLEL):
            self.report("No generator enabled: set mattergen.gen_enabled and/or gnome.enabled in input.yaml")
            return self.exit_codes.ERROR_GENERATIVE_FAILED

        for chemical_system in self.ctx.chemical_systems:
            if settings.MATTERGEN_GEN_ENABLED:
                builder = self._construct_mattergen_gen_builder(chemical_system)
                future = self.submit(builder)
                self.to_context(**{f"{chemical_system}_mattergen": future})

            if settings.GNOME_PARALLEL:
                gbuilder = self._construct_gnome_gen_builder(chemical_system)
                gfuture = self.submit(gbuilder)
                self.to_context(**{f"{chemical_system}_gnome": gfuture})

    def inspect_gen_calcs(self):
        """Best-effort check of each generator branch. Both MatterGen and GNoME
        are optional (toggled in input.yaml); a failed branch only warns here --
        predict_ml_energies fails a system only if no generator yielded any
        structures."""
        for chemical_system in self.ctx.chemical_systems:
            mkey = f"{chemical_system}_mattergen"
            if mkey in self.ctx and not self.ctx[mkey].is_finished_ok:
                self.report(f"Warning: MatterGen branch for {chemical_system} failed")

            gkey = f"{chemical_system}_gnome"
            if gkey in self.ctx and not self.ctx[gkey].is_finished_ok:
                self.report(f"Warning: GNoME branch for {chemical_system} failed")

    def predict_ml_energies(self):
        for chemical_system in self.ctx.chemical_systems:
            output_structures = []

            mkey = f"{chemical_system}_mattergen"
            if mkey in self.ctx and self.ctx[mkey].is_finished_ok:
                try:
                    output_structures.extend(self.ctx[mkey].outputs.output_dict["structures"])
                except Exception:
                    self.report(f"Warning: could not read MatterGen structures for {chemical_system}")

            gkey = f"{chemical_system}_gnome"
            if gkey in self.ctx and self.ctx[gkey].is_finished_ok:
                try:
                    output_structures.extend(self.ctx[gkey].outputs.output_dict["structures"])
                except Exception:
                    self.report(f"Warning: could not read GNoME structures for {chemical_system}")

            if not output_structures:
                self.report(f"No generated structures for {chemical_system}")
                return self.exit_codes.ERROR_GENERATIVE_FAILED

            builder = self._construct_ML_relax_builder(output_structures)
            builder.local_label = Str("relax: {}".format(chemical_system))
            future = self.submit(builder)
            self.to_context(**{f"{chemical_system}_ml_e": future})

    def store_ml_energies(self):
        ml_bulk_model = self.ctx.ml_bulk_model
        chemical_systems = self.ctx.chemical_systems

        failed_ml_e = []
        for chemical_system in chemical_systems:
            wch = self.ctx[f"{chemical_system}_ml_e"]

            if not wch.is_finished_ok:
                self.report(f"Warning: {self.ctx.ml_bulk_model} for {chemical_system} failed")
                failed_ml_e.append(chemical_system)
                continue
            try:
                new_entries = get_output_as_entry(wch)

            except:
                self.report(f"Warning: Failed to store results for {chemical_system}")
                failed_ml_e.append(chemical_system)
                continue

            low_energy_entries = unique_low_energy_chemsys(chemical_system, new_entries,
                                                           EHULL_ML,
                                                           min_n_return=1,
                                                           element_entries=self.ctx.ref_entries)

            if not low_energy_entries:
                self.report(f"Warning: No low-energy generated structures for {chemical_system}")
                failed_ml_e.append(chemical_system)
                continue

            structure_energy_pairs = []
            for entry in low_energy_entries:
                structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))
            add_structures("generated", ml_bulk_model, structure_energy_pairs)
            # DBChemsys status is updated to Ready
            rows = query_by_columns(DBChemsys,{"chemsys": chemical_system})
            if not rows:
                self.report(f"ERROR: {chemical_system} was not found in DBChemsys")
                return self.exit_codes.ERROR_NO_CHEMSYS_FOUND

            row = rows[0]
            update_row(DBChemsys, row.uuid,{"gen_structures": "Ready"})

        if failed_ml_e:
            self.report("ML relaxation failed.")
            return self.exit_codes.ERROR_ML_RELAX_FAILED

    def final_report(self):
        self.report(f"GeneratorWorkChain for {self.ctx.chemical_systems} finished successfully")

    ################################################################################
    @staticmethod
    def _construct_mattergen_gen_builder(chemical_system):
        """MatterGen gen Builder"""
        Workflow = WorkflowFactory("mattergen.base")
        builder = Workflow.get_builder()
        builder.chemical_system = Str(chemical_system)
        builder.code = get_code("MatterGen_generate")
        model, model_path, device = get_model_device("MatterGen_generate")

        builder.job_info = Dict(
                {"job_type": "gen",
                 "model_name": model,
                 "model_path": model_path,
                 "device": device,
                 "energy_above_hull": settings.inputs["MatterGen_generate"]["energy_above_hull"],
                 "batch_size": settings.inputs["MatterGen_generate"]["batch_size"],
                 "num_batches": settings.inputs["MatterGen_generate"]["num_batches"]
                }
        )
        builder.max_iterations = Int(2)
        return builder

    @staticmethod
    def _construct_gnome_gen_builder(chemical_system):
        """GNoME (SAPS) de-novo builder, parallel to MatterGen's gen branch."""
        Workflow = WorkflowFactory("gnome.base")
        builder = Workflow.get_builder()
        builder.chemical_system = Str(chemical_system)
        builder.code = get_code("GNoME")

        ji = dict(settings.inputs["GNoME_generate"])
        ji["model_head"] = ji.get("head")
        screen = ji.get("screen", "none")
        if str(screen).lower() != "none":
            model, model_path, device = get_model_device(screen)
            ji.update({"model_name": model, "model_path": model_path, "device": device})

        builder.job_info = Dict(ji)
        builder.max_iterations = Int(2)
        return builder

    def _construct_ML_relax_builder(self, structures):
        ML_model = self.ctx.ml_bulk_model
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)
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

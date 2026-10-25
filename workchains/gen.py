from aiida.orm import Int, Str, List, Dict, CalcJobNode
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain
from uvsib.db.utils import (add_structures, set_chemsys_status,
                            GEN_READY, GEN_EMPTY, GEN_FAILED, GEN_TERMINAL)
#from uvsib.codes.utils import get_mp_element_structures
from uvsib.workchains.utils import (unique_low_energy_chemsys,
                                    get_output_as_entry,
                                    get_ref_entries,
                                    get_code,
                                    get_model_device)
from uvsib.workflows import settings

EHULL_ML = settings.EHULL_ML
# MatterGen/GNoME parser exit code for an empty output.json (no structures)
_EMPTY_OUTPUT_EXIT = 303

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

        spec.exit_code(300, "ERROR_GENERATIVE_FAILED", message="Generation failed for every chemical system")
        spec.exit_code(304, "ERROR_NO_GENERATOR_ENABLED", message="No generator enabled in input.yaml")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems.get_list()
        self.ctx.ml_bulk_model = self.inputs.ml_bulk_model.value
        self.ctx.status = {}
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
            return self.exit_codes.ERROR_NO_GENERATOR_ENABLED

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
        predict_ml_energies classifies each system from what its branches yielded."""
        for chemical_system in self.ctx.chemical_systems:
            for key, name in self._branch_keys(chemical_system):
                if not self.ctx[key].is_finished_ok:
                    self.report(f"Warning: {name} branch for {chemical_system} failed")

    def predict_ml_energies(self):
        """Submit one ML relaxation per system that produced structures.

        A system with no structures is not an error for the run: it is marked
        Empty if every generator branch ran and produced nothing (parser exit
        ERROR_OUTPUT_INCOMPLETE), and Failed if any branch crashed."""
        for chemical_system in self.ctx.chemical_systems:
            output_structures = []
            crashed = False

            for key, name in self._branch_keys(chemical_system):
                node = self.ctx[key]
                if node.is_finished_ok:
                    try:
                        output_structures.extend(node.outputs.output_dict["structures"])
                    except Exception:
                        self.report(f"Warning: could not read {name} structures for {chemical_system}")
                        crashed = True
                elif not self._branch_found_nothing(node):
                    crashed = True

            if not output_structures:
                status = GEN_FAILED if crashed else GEN_EMPTY
                self._mark(chemical_system, status)
                self.report(f"No generated structures for {chemical_system}: marked {status}")
                continue

            builder = self._construct_ML_relax_builder(output_structures)
            builder.local_label = Str("relax: {}".format(chemical_system))
            future = self.submit(builder)
            self.to_context(**{f"{chemical_system}_ml_e": future})

    def store_ml_energies(self):
        ml_bulk_model = self.ctx.ml_bulk_model

        for chemical_system in self.ctx.chemical_systems:
            key = f"{chemical_system}_ml_e"
            if key not in self.ctx:
                continue  # already marked Empty/Failed in predict_ml_energies
            wch = self.ctx[key]

            if not wch.is_finished_ok:
                self.report(f"Warning: {ml_bulk_model} for {chemical_system} failed")
                self._mark(chemical_system, GEN_FAILED)
                continue
            try:
                new_entries = get_output_as_entry(wch)
            except Exception:
                self.report(f"Warning: Failed to store results for {chemical_system}")
                self._mark(chemical_system, GEN_FAILED)
                continue

            low_energy_entries = unique_low_energy_chemsys(chemical_system, new_entries,
                                                           EHULL_ML,
                                                           min_n_return=1,
                                                           element_entries=self.ctx.ref_entries)

            if not low_energy_entries:
                self.report(f"No low-energy generated structures for {chemical_system}: marked {GEN_EMPTY}")
                self._mark(chemical_system, GEN_EMPTY)
                continue

            structure_energy_pairs = []
            for entry in low_energy_entries:
                structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))
            add_structures("generated", ml_bulk_model, structure_energy_pairs)
            self._mark(chemical_system, GEN_READY)

        status = self.ctx.get("status", {})
        counts = {s: list(status.values()).count(s) for s in GEN_TERMINAL}
        self.report(f"Generation status: {counts}")
        if status and counts[GEN_FAILED] == len(status):
            return self.exit_codes.ERROR_GENERATIVE_FAILED

    def _mark(self, chemical_system, status):
        """Record ``status`` for this run and in DBChemsys."""
        self.ctx.setdefault("status", {})[chemical_system] = status
        if not set_chemsys_status(chemical_system, status):
            self.report(f"Warning: {chemical_system} was not found in DBChemsys")

    def _branch_keys(self, chemical_system):
        """(ctx key, name) of every generator branch submitted for the system."""
        keys = [(f"{chemical_system}_mattergen", "MatterGen"), (f"{chemical_system}_gnome", "GNoME")]
        return [(k, n) for k, n in keys if k in self.ctx]

    @staticmethod
    def _branch_found_nothing(node):
        """True if a failed generator branch ran cleanly but produced no
        structures: every calculation it launched finished with the parser's
        ERROR_OUTPUT_INCOMPLETE (empty output.json), none crashed."""
        calcs = [c for c in node.called if isinstance(c, CalcJobNode)]
        return bool(calcs) and all(c.exit_status == _EMPTY_OUTPUT_EXIT for c in calcs)

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

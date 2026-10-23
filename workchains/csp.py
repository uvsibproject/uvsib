import random
from pymatgen.core import Composition
from aiida.engine import WorkChain
from aiida.orm import Str, List, Dict, Int
from aiida.plugins import DataFactory, WorkflowFactory
from uvsib.workchains.utils import (
        unique_low_energy_comp,
        get_output_as_entry,
        get_ref_entries,
        get_code,
        get_model_device)
from uvsib.db.utils import add_structures
from uvsib.workflows import settings


StructureData = DataFactory('core.structure')

EHULL_ML = settings.EHULL_ML
# seconds reserved between the MH runner's wall-clock budget and the SLURM walltime
MH_WALLTIME_MARGIN = 1800

class CSPWorkChain(WorkChain):
    """WorkChain for Crystal Structure Prediction (CSP)"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("n_csp", valid_type=Int)
        spec.input("n_mh", valid_type=Int)
        spec.input("ml_bulk_model", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_csp,
            cls.inspect_csp_calcs,
            cls.predict_ml_energies,
            cls.collect_ml_energies,
            cls.minimahopping,
            cls.mh_energies,
            cls.relax_mh_structures,
            cls.collect_mh_relax,
            cls.final_step,
            cls.final_report
        )

        spec.exit_code(301, "ERROR_CSP_FAILED", message="CSP calculations failed")
        spec.exit_code(302, "ERROR_ML_RELAX_FAILED", message="ML relaxation failed")
        spec.exit_code(303, "ERROR_MINIMAHOPPING_FAILED", message="MinimaHopping calculations failed")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.n_csp = self.inputs.n_csp.value
        self.ctx.n_mh = self.inputs.n_mh.value
        self.ctx.ml_bulk_model = self.inputs.ml_bulk_model.value
        self.ctx.csp_structures = []
        self.report(f"Launching CSPWorkChain for {self.ctx.chemical_formula}")
        self.ctx.ref_entries, _ = get_ref_entries(self.ctx.chemical_formula, self.ctx.ml_bulk_model)

    def run_csp(self):
        """Run MatterGen CSP, GNoME (SAPS) CSP, and/or DiffCSP in parallel, per
        input.yaml toggles (`mattergen.csp_enabled`, `gnome.enabled`,
        `diffcsp.enabled`); at least one required."""
        self.report("Launching CSP calculations")
        self.ctx.n_csp_jobs = 0

        natom = Composition(self.ctx.chemical_formula).num_atoms
        mattergen_csp_allowed = settings.MATTERGEN_CSP_ENABLED and natom <= 20

        if settings.MATTERGEN_CSP_ENABLED and not mattergen_csp_allowed:
            self.report(
                f"Skipping MatterGen CSP for {self.ctx.chemical_formula}: "
                f"formula has {natom} atoms, above MatterGen's 20-atom CSP limit."
            )

        if mattergen_csp_allowed:
            self.ctx.n_csp_jobs = self.ctx.n_csp
            for i in range(1, self.ctx.n_csp + 1):
                builder = self._construct_mattergen_csp_builder()
                future = self.submit(builder)
                self.to_context(**{f"csp_{i}": future})

        self.ctx.n_gnome = 0
        if settings.GNOME_PARALLEL:
            self.ctx.n_gnome = int(settings.inputs.get("GNoME_CSP", {}).get("num_runs", 1))
            for i in range(1, self.ctx.n_gnome + 1):
                gbuilder = self._construct_gnome_csp_builder()
                self.to_context(**{f"gnome_{i}": self.submit(gbuilder)})

        self.ctx.n_diffcsp = 0
        if settings.DIFFCSP_ENABLED:
            self.ctx.n_diffcsp = int(settings.inputs.get("DiffCSP_CSP", {}).get("num_runs", 1))
            for i in range(1, self.ctx.n_diffcsp + 1):
                dbuilder = self._construct_diffcsp_csp_builder()
                self.to_context(**{f"diffcsp_{i}": self.submit(dbuilder)})

        if self.ctx.n_csp_jobs == 0 and self.ctx.n_gnome == 0 and self.ctx.n_diffcsp == 0:
            self.report("ERROR: no CSP generator enabled (mattergen, gnome, and diffcsp all off)")
            return self.exit_codes.ERROR_CSP_FAILED

    def inspect_csp_calcs(self):
        """Collect structures from MatterGen (primary), GNoME, and DiffCSP (best-effort) CSP"""
        failed_jobs = 0
        for i in range(1, self.ctx.n_csp_jobs + 1):
            csp_wch = self.ctx[f"csp_{i}"]
            if not csp_wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                self.ctx.csp_structures.extend(csp_wch.outputs.output_dict["structures"])
            except:
                failed_jobs += 1

        for i in range(1, self.ctx.n_gnome + 1):
            gnome_wch = self.ctx[f"gnome_{i}"]
            if not gnome_wch.is_finished_ok:
                self.report(f"Warning: GNoME CSP job {i} failed; continuing with other generators")
                continue
            try:
                self.ctx.csp_structures.extend(gnome_wch.outputs.output_dict["structures"])
            except Exception:
                self.report(f"Warning: could not read GNoME CSP structures from job {i}")

        for i in range(1, self.ctx.n_diffcsp + 1):
            diffcsp_wch = self.ctx[f"diffcsp_{i}"]
            if not diffcsp_wch.is_finished_ok:
                self.report(f"Warning: DiffCSP CSP job {i} failed; continuing with other generators")
                continue
            try:
                self.ctx.csp_structures.extend(diffcsp_wch.outputs.output_dict["structures"])
            except Exception:
                self.report(f"Warning: could not read DiffCSP CSP structures from job {i}")

        if not self.ctx.csp_structures:
            self.report("No structure was found. Job stopped.")
            return self.exit_codes.ERROR_CSP_FAILED

        if (self.ctx.n_csp_jobs and failed_jobs / self.ctx.n_csp_jobs > 0.5
                and not settings.GNOME_PARALLEL and not settings.DIFFCSP_ENABLED):
            self.report("Many CSP jobs failed. Job stopped.")
            return self.exit_codes.ERROR_CSP_FAILED

    def predict_ml_energies(self):
        """Predict the energies of the structures with the given ML model"""
        self.ctx.n_main = len(self.ctx.csp_structures)
        builder = self._construct_ML_relax_builder(self.ctx.csp_structures)
        self.to_context(**{"ml_e": self.submit(builder)})

    def collect_ml_energies(self):
        """ML energies"""
        wch = self.ctx.ml_e
        if not wch.is_finished_ok:
            self.report("ML relaxation failed.")
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            new_entries = get_output_as_entry(wch)
        except Exception:
            self.report("ML relaxation failed.")
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        self.ctx.low_energy_entries_csp, _ = unique_low_energy_comp(self.ctx.chemical_formula, new_entries,
                                                                    EHULL_ML, min_n_return=self.ctx.n_mh,
                                                                    element_entries=self.ctx.ref_entries)

    def minimahopping(self):
        """Run MinimaHopping"""
        self.report("Launching MH calculations")
        entries_csp = self.ctx.low_energy_entries_csp
        n_mh = min(len(entries_csp), self.ctx.n_mh)
        selected_entries = random.sample(entries_csp, n_mh)
        for i, entry in enumerate(selected_entries):
            struct = StructureData(pymatgen=entry.structure)
            builder = self._construct_mh_builder(struct)
            future = self.submit(builder)
            self.to_context(**{f"mh_{i}": future})

    def mh_energies(self):
        """Collect MinimaHopping energies"""
        n_mh = min(len(self.ctx.low_energy_entries_csp), self.ctx.n_mh)

        all_mh_entries = []
        failed_jobs = 0
        for i in range(n_mh):
            wch = self.ctx[f"mh_{i}"]
            if not wch.is_finished_ok:
                failed_jobs += 1
                continue
            try:
                new_entries = get_output_as_entry(wch)
                all_mh_entries.extend(new_entries)
            except:
                failed_jobs += 1

        if not all_mh_entries or failed_jobs / n_mh > 0.5:
            self.report("Many MH jobs failed.")
            return self.exit_codes.ERROR_MINIMAHOPPING_FAILED

        self.ctx.mh_structures = [entry.structure.as_dict() for entry in all_mh_entries]

    def relax_mh_structures(self):
        """Re-relax the MH minima with the bulk_relax protocol.

        MinimaHopping keeps minima whose geometry optimization hit opt_max_steps
        without converging, and its force criterion differs from the ASE one of
        relax.py. Relaxing them with the same model / fmax / max_steps as every
        other bulk drops the unconverged ones and keeps the hull consistent.
        """
        self.report(f"Re-relaxing {len(self.ctx.mh_structures)} MH minima with {self.ctx.ml_bulk_model}")
        builder = self._construct_ML_relax_builder(self.ctx.mh_structures)
        builder.local_label = Str("relax MH {}".format(self.ctx.chemical_formula))
        self.to_context(**{"ml_e_mh": self.submit(builder)})

    def collect_mh_relax(self):
        """Select the low-energy MH structures that converged in the re-relaxation"""
        wch = self.ctx.ml_e_mh
        if not wch.is_finished_ok:
            self.report("ML relaxation of the MH minima failed.")
            return self.exit_codes.ERROR_ML_RELAX_FAILED
        try:
            mh_entries = get_output_as_entry(wch)
        except Exception:
            self.report("ML relaxation of the MH minima failed.")
            return self.exit_codes.ERROR_ML_RELAX_FAILED

        n_dropped = len(self.ctx.mh_structures) - len(mh_entries)
        if n_dropped:
            self.report(f"{n_dropped} of {len(self.ctx.mh_structures)} MH minima did not converge "
                        "in the re-relaxation and were dropped.")

        if not mh_entries:
            self.report("Warning: no MH minimum converged; continuing with the CSP structures only.")
            self.ctx.low_energy_entries_mh = []
            return

        self.ctx.low_energy_entries_mh, _ = unique_low_energy_comp(self.ctx.chemical_formula, mh_entries,
                                                                   EHULL_ML,
                                                                   element_entries=self.ctx.ref_entries)

    def final_step(self):
        """Store structures"""
        ml_bulk_model = self.ctx.ml_bulk_model
        all_entries = self.ctx.low_energy_entries_csp + self.ctx.low_energy_entries_mh

        low_energy_entries, _ = unique_low_energy_comp(self.ctx.chemical_formula, all_entries,
                                                       EHULL_ML,
                                                       element_entries=self.ctx.ref_entries)
   
        structure_energy_pairs = []
        for entry in low_energy_entries:
            structure_energy_pairs.append((entry.structure.as_dict(), entry.energy))

        add_structures("csp", ml_bulk_model, structure_energy_pairs)

    def final_report(self):
        """Final report"""
        self.report(f"CSPWorkChain for {self.ctx.chemical_formula} finished successfully")

####################################################################
    def _construct_mattergen_csp_builder(self):
        Workflow = WorkflowFactory("mattergen.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("MatterGen_CSP")
        _, model_path, _ = get_model_device("MatterGen_CSP")
        builder.job_info = Dict(
            {
                "job_type": "csp",
                "model_path": model_path,
                "batch_size": settings.inputs["MatterGen_CSP"]["batch_size"],
                "num_batches": settings.inputs["MatterGen_CSP"]["num_batches"],
            }
        )
        builder.max_iterations = Int(2)
        return builder

    def _construct_gnome_csp_builder(self):
        """GNoME (SAPS) CSP builder, parallel to MatterGen's csp branch."""
        Workflow = WorkflowFactory("gnome.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("GNoME")

        ji = dict(settings.inputs["GNoME_CSP"])
        ji["model_head"] = ji.get("head")
        screen = ji.get("screen", "none")
        if str(screen).lower() != "none":
            model, model_path, device = get_model_device(screen)
            ji.update({"model_name": model, "model_path": model_path, "device": device})

        builder.job_info = Dict(ji)
        builder.max_iterations = Int(2)
        return builder

    def _construct_diffcsp_csp_builder(self):
        """DiffCSP (composition-conditioned) CSP builder, parallel to
        MatterGen's and GNoME's csp branches. `repo_path` is an explicit,
        machine-specific path under config.yaml's codes.DiffCSP_CSP (the
        DiffCSP checkout); `model_path` is resolved the same way as
        MatterGen/GNoME, via configs["models"]["DiffCSP_CSP"] -- see
        docs/diffcsp_generation.md."""
        Workflow = WorkflowFactory("diffcsp.csp")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.code = get_code("DiffCSP_CSP")
        _, model_path, _ = get_model_device("DiffCSP_CSP")

        codes_cfg = settings.configs["codes"]["DiffCSP_CSP"]
        ji = dict(settings.inputs["DiffCSP_CSP"])
        ji.update({
            "repo_path": codes_cfg["repo_path"],
            "model_path": model_path,
        })

        builder.job_info = Dict(ji)
        builder.max_iterations = Int(2)
        return builder

    def _construct_ML_relax_builder(self, structures):
        """
        General builder for structure optimization with an ML model
        """
        ML_model = self.ctx.ml_bulk_model
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structures)
        builder.code = get_code(ML_model)
        builder.local_label = Str("relax {}".format(self.ctx.chemical_formula))
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

    def _construct_mh_builder(self, struct):
        Workflow = WorkflowFactory("minimahopping")
        builder = Workflow.get_builder()
        builder.structure = struct
        builder.code = get_code("MinimaHopping")
        builder.this_label = '{}'.format(self.ctx.chemical_formula)
        model, model_path, device = get_model_device(settings.inputs["MinimaHopping"]["model"])

        # MH runner wall-clock budget: the SLURM walltime of the MinimaHopping
        # code minus MH_WALLTIME_MARGIN, so the runner stops and writes
        # output.json well before the scheduler kills the job.
        walltime = int(settings.configs["codes"]["MinimaHopping"]["job_script"]["time"])
        max_seconds = walltime - MH_WALLTIME_MARGIN
        if max_seconds <= 0:
            raise ValueError(
                f"codes.MinimaHopping.job_script.time={walltime}s in config.yaml must exceed "
                f"the {MH_WALLTIME_MARGIN}s MinimaHopping wall-clock margin.")

        job_info = {
            "job_type": "hopping",
            "ML_model": settings.inputs["MinimaHopping"]["model"],
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs["MinimaHopping"]["head"],
            "device": device,
            "mh_steps": settings.inputs["MinimaHopping"]["mh_steps"],
            "fmax": settings.inputs["MinimaHopping"]["fmax"],
            "max_seconds": max_seconds
        }

        builder.job_info = Dict(job_info)
        return builder

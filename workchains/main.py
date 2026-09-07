import os
from aiida.orm import Str, List, Dict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain, if_, while_
from aiida_pythonjob import PythonJob, prepare_pythonjob_inputs
from uvsib.db.tables import DBComposition, DBSurfaceMLAdsorbate
from uvsib.db.utils import (update_row, query_by_columns, update_step_status_path,
                            update_json_path)
from uvsib.workchains.pythonjob_inputs import wait_sleep
from uvsib.workflows import settings

_PD_VERIFICATION = settings._PD_VERIFICATION


def _ads_status(step_status, reaction, reaction_path):
    """Per-(reaction, pathway) adsorbates status from the nested step_status.

    ``step_status["adsorbates"]`` is keyed reaction -> reaction_path -> state so
    that catalyst chains running in parallel for one composition (e.g. CO2RR and
    NOXRR) do not share a single flat 'adsorbates' flag.
    """
    return ((step_status or {}).get("adsorbates") or {}).get(
        reaction, {}).get(reaction_path)


def _akmc_status(step_status, reaction, reaction_path):
    """Per-(reaction, pathway) AKMC status from the nested step_status."""
    return ((step_status or {}).get("akmc") or {}).get(
        reaction, {}).get(reaction_path)


def _report_status(step_status, reaction, reaction_path):
    """Per-(reaction, pathway) pipeline_report status from the nested step_status."""
    return ((step_status or {}).get("pipeline_report") or {}).get(
        reaction, {}).get(reaction_path)


class MainWorkChain(WorkChain):
    """ Main WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("chemical_systems", valid_type=List)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)
        spec.input("similarities", valid_type=Dict)
        spec.input('sqs', valid_type=Dict)

        spec.outline(
            cls.setup,
            if_(cls.should_run_pd_ml)(
                while_(cls.should_wait_pd_ml)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.pd_ml,
                cls.inspect_pd_ml
            ),
            if_(cls.should_run_pd_verification)(
                while_(cls.should_wait_pd_ver)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.pd_verification,
                cls.inspect_pd_verification
            ),
            if_(cls.should_run_synthesizability)(
                while_(cls.should_wait_synthesizability)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.synthesizability,
                cls.inspect_synthesizability
            ),
            if_(cls.should_run_sqs)(
                while_(cls.should_wait_sqs)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.sqs,
                cls.inspect_sqs
            ),
            if_(cls.should_run_surface_builder)(
                while_(cls.should_wait_surface_builder)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.surface_builder,
                cls.inspect_surface_builder
            ),
            if_(cls.should_run_adsorbates)(
                while_(cls.should_wait_adsorbates)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.adsorbates,
                cls.inspect_adsorbates
            ),
            if_(cls.should_run_akmc)(
                while_(cls.should_wait_akmc)(
                    cls.wait_sleep,
                    cls.check_pythonjob_sleep
                ),
                cls.akmc,
                cls.inspect_akmc
            ),
            if_(cls.should_run_pipeline_report)(
                cls.pipeline_report
            )
        )

        spec.exit_code(300,"ERROR_CALCULATION_FAILED", message="A sub-WorkChain did not finish successfully")
        spec.exit_code(305,"ERROR_COMPOSITION_MISSING", message="The composition is missing in DBComposition")


    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.chemical_systems = self.inputs.chemical_systems
        self.ctx.reaction = self.inputs.reaction
        self.ctx.reaction_path = self.inputs.reaction_path
        self.ctx.ml_bulk_model = settings.inputs['bulk_relax']['model'] # not an input of the workchain
        rows = query_by_columns(DBComposition,{"composition": self.ctx.chemical_formula})
        if not rows:
            self.report(f"ERROR: no DBComposition row for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_COMPOSITION_MISSING
        self.ctx.dbcomposition_row = rows[0]
        self.ctx.composition_missing = False
        self.ctx.similarities = self.inputs.similarities.value
        self.ctx.sqs_request = self.inputs.sqs.get_dict()
        self.ctx.sqs = bool(self.ctx.sqs_request)   # non-empty request -> SQS run
        self.report(f"Running MainWorkChain for {self.ctx.chemical_formula}: "
                    f"reaction {self.ctx.reaction.value} path {self.ctx.reaction_path.value}")


    def _fresh_step_status(self):
        """
        Re-query the composition row so the shared-step gates observe a
        sibling chain's transitions.
        """
        rows = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})
        if not rows:
            self.ctx.composition_missing = True
            return {}

        row = rows[0]
        self.ctx.composition_missing = False
        self.ctx.dbcomposition_row = row
        return row.step_status or {}

    def _step_done(self, step):
        """Return True if a shared composition-level step is already done."""
        return self._fresh_step_status().get(step) == "Done"

    def should_run_pd_ml(self):
        """Check whether should run PhaseDiagramML"""
        if self.ctx.sqs:
            return False
        step_status = self._fresh_step_status().get("pd_ml")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_pd_verification(self):
        """Check whether should run PDVerification"""
        if self.ctx.sqs:
            return False
        if not _PD_VERIFICATION:
            return False
        step_status = self._fresh_step_status().get("pd_verification")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_sqs(self):
        """Check whether should run PhaseDiagramML"""
        if not self.ctx.sqs:
            return False
        step_status = self._fresh_step_status().get("sqs")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_synthesizability(self):
        """Check whether should run SynthesizabilityWorkChain"""
        if not settings.SYNTH_ENABLED or self.ctx.sqs:
            return False
        step_status = self._fresh_step_status().get("synthesizability")
        if step_status in ["Done"]:
            return False
        return True

    def should_run_surface_builder(self):
        """Check whether should run SurfaceBuilder"""
        if settings.SOFT_STOP_BEFORE_SURFACE:
            self.report("Soft stop (soft_stop.before_surface_builder): ending before the "
                        "surface builder starts; generation/synthesizability stages are complete now.")
            return False
        surface_builder_step_status = self._fresh_step_status().get("surface_builder")
        if surface_builder_step_status in ["Done"]:
            return False
        return True

    def should_run_adsorbates(self):
        """Check whether should run Adsorbates for THIS (reaction, pathway)"""
        if settings.SOFT_STOP_BEFORE_SURFACE:
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        # Re-query for a fresh status: the setup-time row snapshot would not see
        # a sibling chain's transition. Status is per-(reaction, pathway).
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if _ads_status(comp_row.step_status, reaction, reaction_path) in ["Done"]:
            row = query_by_columns(DBSurfaceMLAdsorbate, {"composition": self.ctx.chemical_formula,
                                                          "reaction": reaction,
                                                          "reaction_path": reaction_path})
            if row:
                return False
        return True

    def should_run_akmc(self):
        """Check whether should run AKMC for THIS (reaction, pathway)."""
        if not settings.AKMC_ENABLED:
            return False
        if settings.SOFT_STOP_BEFORE_SURFACE:
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if _akmc_status(comp_row.step_status, reaction, reaction_path) in ["Done"]:
            return False
        ads_rows = query_by_columns(DBSurfaceMLAdsorbate, {
            "composition": self.ctx.chemical_formula,
            "reaction": reaction,
            "reaction_path": reaction_path,
        })
        return bool(ads_rows)

    def should_run_pipeline_report(self):
        """Check whether the post-pipeline report should be (re)generated for
        THIS (reaction, pathway) -- skipped once a sibling MainWorkChain has
        already generated it."""
        if settings.SOFT_STOP_BEFORE_SURFACE:
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if _report_status(comp_row.step_status, reaction, reaction_path) in ["Done"]:
            return False
        return True

    def should_wait_pd_ml(self):
        """Should wait for another running WorkChain"""
        pd_ml_step_status = self._fresh_step_status().get("pd_ml")
        if pd_ml_step_status in ["Running"]:
            self.ctx.sts = "phase diagram"
            return True
        return False

    def should_wait_pd_ver(self):
        """Should wait for another running WorkChain"""
        pd_ver_step_status = self._fresh_step_status().get("pd_verification")
        if pd_ver_step_status in ["Running"]:
            self.ctx.sts = "phase diagram verification"
            return True
        return False

    def should_wait_sqs(self):
        """Should wait for another running WorkChain"""
        if not self.ctx.sqs:
            return False
        step_status = self._fresh_step_status().get("sqs")
        if step_status in ["Running"]:
            self.ctx.sts = "sqs"
            return True
        return False

    def should_wait_synthesizability(self):
        """Should wait for another running WorkChain"""
        step_status = self._fresh_step_status().get("synthesizability")
        if step_status in ["Running"]:
            self.ctx.sts = "synthesizability"
            return True
        return False

    def should_wait_surface_builder(self):
        """Should wait for another running WorkChain"""
        surface_builder_step_status = self._fresh_step_status().get("surface_builder")
        if surface_builder_step_status in ["Running"]:
            self.ctx.sts = "surface builder"
            return True
        return False

    def should_wait_adsorbates(self):
        """Wait only if this (reaction, pathway) is running elsewhere.
        Different reactions/pathways (e.g. CO2RR vs NOXRR) are independent work
        and must not block each other; only a duplicate of the same triple does.
        """
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if _ads_status(comp_row.step_status, reaction, reaction_path) in ["Running"]:
            self.ctx.sts = f"adsorbates:{reaction}:{reaction_path}"
            return True
        return False

    def should_wait_akmc(self):
        """Wait only if this (reaction, pathway) AKMC is running elsewhere."""
        if not settings.AKMC_ENABLED:
            return False
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        comp_row = query_by_columns(DBComposition, {"composition": self.ctx.chemical_formula})[0]
        if _akmc_status(comp_row.step_status, reaction, reaction_path) in ["Running"]:
            self.ctx.sts = f"akmc:{reaction}:{reaction_path}"
            return True
        return False

    def wait_sleep(self):
        """Wait until the other workchain for this composition ends"""
        self.report(f"Waiting for a similar WorkChain ({self.ctx.sts})")
        inputs = prepare_pythonjob_inputs(wait_sleep, function_inputs={}, computer="localhost")
        future = self.submit(PythonJob, inputs=inputs)
        self.to_context(**{"pyjob_sleep": future})

    def check_pythonjob_sleep(self):
        """Inspect PythonJob"""
        calculation = self.ctx["pyjob_sleep"]
        if not calculation.is_finished_ok:
            self.report("PythonJob failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def pd_ml(self):
        """Running PhaseDiagramML WorkChain"""
        # Re-check after waiting: another MainWorkChain may have completed pd_ml.
        if self._step_done("pd_ml"):
            self.report(
                f"Skipping PhaseDiagramML WorkChain for {self.ctx.chemical_formula}: "
                "pd_ml was completed by another WorkChain."
            )
            return

        if self.ctx.composition_missing:
            self.report(f"ERROR: no DBComposition row for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_COMPOSITION_MISSING


        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"pd_ml": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_pd_ml_builder()
        future = self.submit(builder)
        self.to_context(**{"pd_ml": future})

    def inspect_pd_ml(self):
        """Inspecting PhaseDiagramML WorkChain"""
        # return if WorkChain was not set
        if "pd_ml" not in self.ctx:
            return

        pd_ml_wch = self.ctx.pd_ml
        row = self.ctx.dbcomposition_row
        if not pd_ml_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"pd_ml": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("PhaseDiagramML WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"pd_ml": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})

    def pd_verification(self):
        """Running PDVerificationWorkChain"""
        # Re-check after waiting: another MainWorkChain may have completed pd_verification.
        if self._step_done("pd_verification"):
            self.report(
                f"Skipping PDVerification WorkChain for {self.ctx.chemical_formula}: "
                "it was completed by another WorkChain."
            )
            return

        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"pd_verification": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_pd_verification_builder()
        future = self.submit(builder)
        self.to_context(**{"pdverification": future})

    def inspect_pd_verification(self):
        """Inspecting PDVerificationWorkChain"""
        # return if WorChain was not set
        if "pdverification" not in self.ctx:
            return

        pd_ver_wch = self.ctx.pdverification
        row = self.ctx.dbcomposition_row

        if not pd_ver_wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"pd_verification": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed","step_status": row.step_status})
            self.report("PDVerification WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"pd_verification": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running","step_status": row.step_status})

    def synthesizability(self):
        """Running SynthesizabilityWorkChain (classify all generated structures)"""
        # Re-check after waiting: another MainWorkChain may have completed synthesizability.
        if self._step_done("synthesizability"):
            self.report(
                f"Skipping Synthesizability WorkChain for {self.ctx.chemical_formula}: "
                "it was completed by another WorkChain."
            )
            return

        row = self.ctx.dbcomposition_row
        row.step_status.update({"synthesizability": "Running"})
        update_row(DBComposition, row.uuid, {"status": "Running", "step_status": row.step_status})
        builder = self._construct_synthesizability_builder()
        future = self.submit(builder)
        self.to_context(**{"synthesizability": future})

    def inspect_synthesizability(self):
        """Inspecting SynthesizabilityWorkChain"""
        # return if WorChain was not set
        if "synthesizability" not in self.ctx:
            return

        wch = self.ctx.synthesizability
        row = self.ctx.dbcomposition_row
        if not wch.is_finished_ok:
            row.step_status.update({"synthesizability": "Failed"})
            update_row(DBComposition, row.uuid, {"status": "Failed", "step_status": row.step_status})
            self.report("Synthesizability WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        row.step_status.update({"synthesizability": "Done"})
        update_row(DBComposition, row.uuid, {"status": "Running", "step_status": row.step_status})

    def sqs(self):
        """Running SQS WorkChain"""
        # Re-check after waiting: another MainWorkChain may have completed pd_ml.
        if self._step_done("sqs"):
            self.report(
                f"Skipping SQS WorkChain for {self.ctx.chemical_formula}: "
                "it was completed by another WorkChain."
            )
            return

        row = self.ctx.dbcomposition_row
        # update row status in DBComposition table
        row.step_status.update({"sqs": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_sqs_builder(self.ctx.sqs_request)
        future = self.submit(builder)
        self.to_context(**{"sqs": future})

    def inspect_sqs(self):
        """Inspecting SQS WorkChain"""
        # return if WorChain was not set
        if "sqs" not in self.ctx:
            return

        wch = self.ctx.sqs
        row = self.ctx.dbcomposition_row
        if not wch.is_finished_ok:
            # update row status in DBComposition table
            row.step_status.update({"sqs": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("SQS WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # update row status in DBComposition table
        row.step_status.update({"sqs": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})

    def surface_builder(self):
        """Running SurfaceBuilderWorkChain"""
        # Re-check after waiting: another MainWorkChain may have completed pd_ml.
        if self._step_done("surface_builder"):
            self.report(
                f"Skipping SurfaceBuilder WorkChain for {self.ctx.chemical_formula}: "
                "it completed by another WorkChain."
            )
            return

        row = self.ctx.dbcomposition_row
        row.step_status.update({"surface_builder": "Running"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})
        builder = self._construct_surface_builder()
        future = self.submit(builder)
        self.to_context(**{"surface_builder": future})

    def inspect_surface_builder(self):
        """Inspecting SurfaceBuilderWorkChain"""
        # return if WorChain was not set
        if "surface_builder" not in self.ctx:
            return

        wch = self.ctx.surface_builder
        row = self.ctx.dbcomposition_row
        if not wch.is_finished_ok:
            row.step_status.update({"surface_builder": "Failed"})
            update_row(DBComposition, row.uuid,{"status": "Failed", "step_status": row.step_status})
            self.report("SurfaceBuilder WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        row.step_status.update({"surface_builder": "Done"})
        update_row(DBComposition, row.uuid,{"status": "Running", "step_status": row.step_status})

    def adsorbates(self):
        """Running AdsorbatesWorkChain"""
        row = self.ctx.dbcomposition_row
        path = ["adsorbates", self.ctx.reaction.value, self.ctx.reaction_path.value]
        # Atomic per-(reaction, pathway) write -- does NOT overwrite siblings'
        # adsorbates keys (a read-modify-write of the whole JSONB would).
        update_step_status_path(DBComposition, row.uuid, path, "Running")
        update_row(DBComposition, row.uuid, {"status": "Running"})
        builder = self._construct_adsorbates_builder()
        future = self.submit(builder)
        self.to_context(**{"adsorbates": future})

    def inspect_adsorbates(self):
        """Inspecting SurfaceBuilderWorkChain"""
        # return if WorChain was not set
        if "adsorbates" not in self.ctx:
            return

        wch = self.ctx.adsorbates
        row = self.ctx.dbcomposition_row
        path = ["adsorbates", self.ctx.reaction.value, self.ctx.reaction_path.value]
        if not wch.is_finished_ok:
            update_step_status_path(DBComposition, row.uuid, path, "Failed")
            update_row(DBComposition, row.uuid, {"status": "Failed"})
            self.report("Adsorbates WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        update_step_status_path(DBComposition, row.uuid, path, "Done")
        status = "Running" if settings.AKMC_ENABLED else "Done"
        update_row(DBComposition, row.uuid, {"status": status})

    def akmc(self):
        """Running AKMCWorkChain"""
        # Re-check after waiting: another MainWorkChain may have completed akmc.
        if self._step_done("akmc"):
            self.report(
                f"Skipping AKMC WorkChain for {self.ctx.chemical_formula}: "
                "it  was completed by another WorkChain."
            )
            return

        row = self.ctx.dbcomposition_row
        path = ["akmc", self.ctx.reaction.value, self.ctx.reaction_path.value]
        update_step_status_path(DBComposition, row.uuid, path, "Running")
        update_row(DBComposition, row.uuid, {"status": "Running"})
        builder = self._construct_akmc_builder()
        future = self.submit(builder)
        self.to_context(**{"akmc": future})

    def inspect_akmc(self):
        """Inspecting AKMCWorkChain"""
        # return if WorChain was not set
        if "akmc" not in self.ctx:
            return

        wch = self.ctx.akmc
        row = self.ctx.dbcomposition_row
        path = ["akmc", self.ctx.reaction.value, self.ctx.reaction_path.value]
        if not wch.is_finished_ok:
            update_step_status_path(DBComposition, row.uuid, path, "Failed")
            update_row(DBComposition, row.uuid, {"status": "Failed"})
            self.report("AKMC WorkChain failed")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        update_step_status_path(DBComposition, row.uuid, path, "Done")
        update_row(DBComposition, row.uuid, {"status": "Done"})

    def pipeline_report(self):
        """Generate and persist the post-pipeline report (bulk/surface/
        reaction-path figures + tables, then an HTML page presenting them)
        for THIS (reaction, pathway), now that PhaseDiagramMLWorkChain /
        SurfaceBuilderWorkChain / AdsorbatesWorkChain (and AKMCWorkChain, if
        enabled) have all written the rows ``pipeline_report.py`` joins.
        Output goes to one folder per (composition, reaction, reaction_path)
        under ``settings.REPORTS_DIR`` (a fixed directory next to the
        ``uvsib`` package, NOT the per-run ``settings.run_dir``) so
        parallel/repeated runs never collide or overwrite each other, and
        reports always land in the same predictable place regardless of
        which run directory produced them. ``render_html_report()`` writes its
        own "raw_data.json" (tables + full bulk/surface structures) next to
        "report.html", so this step does not separately dump a summary.json
        -- that would just be a lighter, easily-stale duplicate of it."""
        import uvsib.workchains.pipeline_report as pr

        row = self.ctx.dbcomposition_row
        reaction = self.ctx.reaction.value
        reaction_path = self.ctx.reaction_path.value
        path = ["pipeline_report", reaction, reaction_path]
        update_step_status_path(DBComposition, row.uuid, path, "Running")

        folder = f"{self.ctx.chemical_formula}_{reaction}_{reaction_path}"
        output_dir = os.path.join(settings.REPORTS_DIR, folder)
        os.makedirs(output_dir, exist_ok=True)

        try:
            summaries = pr.report(self.ctx.chemical_formula, reaction, reaction_path,
                                   plot_dir=output_dir)
            pr.render_html_report(self.ctx.chemical_formula, reaction, reaction_path,
                                   summaries=summaries,
                                   output_path=os.path.join(output_dir, "report.html"))
        except Exception:
            update_step_status_path(DBComposition, row.uuid, path, "Failed")
            raise

        # Record the browser URL of the report BEFORE marking the step Done, so
        # any update_dbfrontend() sweep that sees "Done" also finds the link to
        # copy into the db_frontend row's ``result`` for this (reaction, path).
        report_url = f"{settings.REPORTS_URL_PREFIX}/{folder}/report.html"
        try:
            update_json_path(DBComposition, row.uuid, "attributes",
                             ["reports", reaction, reaction_path], report_url)
        except Exception as exc:
            self.report(f"WARNING: could not record report URL for "
                        f"{self.ctx.chemical_formula} ({reaction}/{reaction_path}): {exc}")

        update_step_status_path(DBComposition, row.uuid, path, "Done")
        self.report(f"Pipeline report for {self.ctx.chemical_formula} "
                    f"({reaction}/{reaction_path}) written to {output_dir}")

    ################################################################################
    def _construct_pd_ml_builder(self):
        """Build PhaseDiagramML WorkChain builder"""
        PhaseDiagramMLWorkChain = WorkflowFactory("phasediagram")
        builder = PhaseDiagramMLWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.chemical_systems = self.ctx.chemical_systems
        builder.ml_bulk_model = Str(self.ctx.ml_bulk_model)
        return builder

    def _construct_pd_verification_builder(self):
        """Build PDVerification WorkChain builder"""
        PDVerificationWorkChain = WorkflowFactory("pdverification")
        builder = PDVerificationWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

    def _construct_synthesizability_builder(self):
        """Build Synthesizability WorkChain builder"""
        SynthesizabilityWorkChain = WorkflowFactory("synthesizability")
        builder = SynthesizabilityWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

    def _construct_surface_builder(self):
        """SurfaceBuilder WorkChain builder"""
        SurfaceBuilderWorkChain = WorkflowFactory("surfacebuilder")
        builder = SurfaceBuilderWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        return builder

    def _construct_adsorbates_builder(self):
        """Adsorbates WorkChain builder"""
        AdsorbatesWorkChain = WorkflowFactory("adsorbates")
        builder = AdsorbatesWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.reaction = self.ctx.reaction
        builder.reaction_path = self.ctx.reaction_path
        return builder

    def _construct_akmc_builder(self):
        """AKMC WorkChain builder"""
        AKMCWorkChain = WorkflowFactory("akmc")
        builder = AKMCWorkChain.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.reaction = self.ctx.reaction
        builder.reaction_path = self.ctx.reaction_path
        return builder

    def _construct_sqs_builder(self, request):
        """SQS WorkChain builder.

        The ``request`` payload (parent structure, sublattices,
        composition_grid, surfaces, defects) is the ``sqs_request`` dict passed
        through the submission entry (see run_dir/run.py), threaded here via
        ``ctx.sqs_request``. Optional ``mu_O2`` (eV per O2 molecule,
        MLIP-relaxed) and ``functional`` (elemental reference set for the bulk
        hull) are keys inside that same request dict.
        """
        WorkChain = WorkflowFactory("sqs")
        builder = WorkChain.get_builder()
        builder.request = Dict(dict=request)
        builder.local_label = Str('SQS for {}'.format(request['parent_label']))
        # if "mu_O2" in settings.inputs['SQS']:
        #     from aiida.orm import Float
        #     builder.mu_O2 = Float(float(settings.inputs['SQS']["mu_O2"]))
        return builder

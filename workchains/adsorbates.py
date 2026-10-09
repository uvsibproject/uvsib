from ase.io import jsonio
from aiida.engine import WorkChain, if_
from aiida.plugins import WorkflowFactory
from aiida.orm import Str, List, Dict
from uvsib.db.tables import DBSurface
from uvsib.db.utils import add_surface_ml_adsorbate
from uvsib.db.utils import query_by_columns
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workchains.oer import calculate_oer_overpotential
from uvsib.workchains.co2rr import calculate_co2rr_overpotential
from uvsib.workchains.noxrr import calculate_noxrr_overpotential
from uvsib.workchains.cer import calculate_cer_overpotential
from uvsib.workchains.her import calculate_her_overpotential
from uvsib.workchains.nrr import calculate_nrr_overpotential
from uvsib.workchains.orr import calculate_orr_overpotential
from uvsib.workflows import settings

MAX_NUM_ADS = settings.MAX_NUM_ADS

# reaction -> (overpotential calculator, eta storage threshold in V); shared
# with AdsorptionUncertaintyWorkChain so committee models go through exactly
# the same CHE bookkeeping
REACTION_FUNCTIONS = {
    "OER": (calculate_oer_overpotential, 2.0),
    "CO2RR": (calculate_co2rr_overpotential, 2.0),
    "CER": (calculate_cer_overpotential, 2.0),
    "NRR": (calculate_nrr_overpotential, 2.0),
    "NOXRR": (calculate_noxrr_overpotential, 2.0),
    "HER": (calculate_her_overpotential, 2.0),
    "ORR": (calculate_orr_overpotential, 2.0)
}

def get_structure_uuid_surface_id(chemical_formula):
    """
    Return (structure_uuid, surface_id) tuples for all DBSurface entries 
    with the given composition
    """
    rows = query_by_columns(DBSurface, {"composition": chemical_formula})
    return [(row.structure_uuid, row.id) for row in rows]

class AdsorbatesWorkChain(WorkChain):
    """Adsorbates WorkChain"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)
        spec.input("reaction", valid_type=Str)
        spec.input("reaction_path", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_adsorbs,
            cls.inspect_adsorbs,
            cls.store_results_ml,
            if_(cls.should_run_uncertainty)(
                cls.run_uncertainty,
                cls.inspect_uncertainty
            ),
            cls.final_report
        )

        spec.exit_code(300,"ERROR_CALCULATION_FAILED", message="The calculation did not finish successfully")
        spec.exit_code(301,"ERROR_NO_SURFACE_FOUND", message="No stable surface was found for the given formula.")
        spec.exit_code(302,"NO_CANDIDATES_WITHIN_ETA_LIMIT", message="No candidates we within the given eta limit")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.ctx.reaction = self.inputs.reaction.value
        self.ctx.reaction_path = self.inputs.reaction_path.value
        self.ctx.structure_surface_rows = get_structure_uuid_surface_id(self.ctx.chemical_formula)
        self.ctx.selected_surfaces = []
        self.ctx.ml_results = {}
        self.ctx.candidates = 0
        self.ctx.relaxation_results = {}
        self.ctx.adsorption_sets = {}
        self.report(f"Running Adsorbates WorkChain for {self.ctx.chemical_formula}. "
                    f"Reaction: {self.ctx.reaction}, reaction_path: {self.ctx.reaction_path}")
        if not self.ctx.structure_surface_rows:
            self.report(f"ERROR: no stable surface was found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_SURFACE_FOUND

    def run_adsorbs(self):
        """Run Adsorbates WorkChain: up to MAX_NUM_ADS most stable surfaces per bulk uuid."""
        by_uuid = {}
        for structure_uuid, surface_id in self.ctx.structure_surface_rows:
            slab_row = query_by_columns(DBSurface, {"id": surface_id})[0]
            sfe = slab_row.formation_energy if slab_row.formation_energy is not None else float("inf")
            by_uuid.setdefault(str(structure_uuid), []).append((sfe, slab_row, surface_id))

        for uid, rows in by_uuid.items():
            rows.sort(key=lambda x: x[0])
            considered = rows[:MAX_NUM_ADS]
            self.report(f"uuid={uid}: {len(considered)} surface(s) considered "
                        f"(of {len(rows)} available)")
            for sfe, row, fid in considered:
                self.ctx.selected_surfaces.append((f"{uid}", f"{fid}"))
                builder = self._construct_adsorbate_builder(row.slab, self.ctx.reaction, self.ctx.reaction_path)
                future = self.submit(builder)
                self.to_context(**{f"{uid}_{fid}": future})

    def inspect_adsorbs(self):
        """Inspect Adsorbates WorkChain"""
        failed = []
        for uid, fid in self.ctx.selected_surfaces:
            ads_wch = self.ctx[f"{uid}_{fid}"]
            if not ads_wch.is_finished_ok:
                failed.append(f"{uid}_{fid} (exit status: {ads_wch.exit_status})")
                continue

            output_dict = ads_wch.outputs.output_dict
            self.ctx.ml_results[f"{str(uid)}_{fid}"] = output_dict["structures"]

        if len(failed) == len(self.ctx.selected_surfaces):
            self.report("ERROR: all sub-workflows failed - no results to process.")
            return self.exit_codes.ERROR_CALCULATION_FAILED
        if failed:
            self.report(f"WARNING: {len(failed)} sub-workflow(s) failed.")

    def store_results_ml(self):
        """Store ML results """
        reaction_map = REACTION_FUNCTIONS

        if self.ctx.reaction not in reaction_map:
            self.report(f"The reaction {self.ctx.reaction} is not known")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # self.report(f"Storing {self.ctx.reaction} ML results.")
        calc_method, eta_threshold = reaction_map[self.ctx.reaction]
        slab_cache = {}
        for parent_key, adsorption_sets in self.ctx.ml_results.items():
            uuid_str, surface_id = parent_key.split("_")
            if surface_id not in slab_cache:
                slab_cache[surface_id] = query_by_columns(DBSurface, {"id": surface_id})[0]
            slab_row = slab_cache[surface_id]
            miller_index = slab_row.slab["miller_index"]
            for adsorb_set in adsorption_sets:
                site_type = adsorb_set["site_type"]
                ads_coord = adsorb_set["ads_coord"]
                repeat = adsorb_set["repeat"]
                energy_set = {}
                for ads_json in adsorb_set["structures"]:
                    adsorbed = jsonio.decode(ads_json)
                    energy_set[adsorbed.info["adsorbate"]] = adsorbed.info['{}_energy'.format(str(settings.inputs['adsorbates']['model']).lower())]

                try:
                    eta, dG_steps, dG_cumulative = calc_method(energy_set, self.ctx.reaction_path)
                except KeyError as missing:
                    # A pathway intermediate is absent from the relaxed set -- e.g.
                    # a fragile dimer like *N2O2 that dissociated during relaxation
                    # and was dropped by the adsorbate validator. Skip THIS
                    # candidate instead of crashing the whole workchain; other
                    # sites/surfaces (and other pathways) are unaffected.
                    self.report(
                        f"Skipping {self.ctx.reaction}/{self.ctx.reaction_path} on "
                        f"surface {surface_id} ({site_type}): missing intermediate "
                        f"{missing.args[0]} in the relaxed set "
                        f"(have: {sorted(energy_set)}).")
                    continue

                if eta > eta_threshold:
                    continue

                self.ctx.candidates += 1
                add_surface_ml_adsorbate(existing_uuid=uuid_str, surf_id=surface_id, surface_miller_index=miller_index,
                                         comp=self.ctx.chemical_formula,
                                         react=self.ctx.reaction, react_path=self.ctx.reaction_path,
                                         site_type=site_type, ads_coord=ads_coord, repeat=repeat,
                                         e=eta, dG_steps=dG_steps, dG_cumulative=dG_cumulative,
                                         ad_set=adsorb_set)

    def should_run_uncertainty(self):
        """Committee eta uncertainty (AdsorptionUncertaintyWorkChain). Opt-in via
        ``settings.ADSORPTION_UNCERTAINTY_ENABLED``; needs stored candidates."""
        return settings.ADSORPTION_UNCERTAINTY_ENABLED and self.ctx.candidates > 0

    def run_uncertainty(self):
        """Submit AdsorptionUncertaintyWorkChain for the stored candidates."""
        Workflow = WorkflowFactory("adsorptionuncertainty")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        builder.reaction = Str(self.ctx.reaction)
        builder.reaction_path = Str(self.ctx.reaction_path)
        self.to_context(**{"uncertainty": self.submit(builder)})

    def inspect_uncertainty(self):
        """Advisory stage: never fail the adsorbates workchain on the uncertainty analysis."""
        wch = self.ctx.uncertainty
        if not wch.is_finished_ok:
            self.report(f"AdsorptionUncertaintyWorkChain did not finish OK (exit {wch.exit_status}); "
                        "continuing without eta uncertainty.")
            return
        self.report("AdsorptionUncertaintyWorkChain finished; uncertainty written for the stored candidates.")

    def final_report(self):
        """Final report"""
        if self.ctx.candidates == 0:
            self.report(f"AdsorbatesWorkChain for {self.ctx.chemical_formula}: no candidates below eta threshold.")
            return self.exit_codes.NO_CANDIDATES_WITHIN_ETA_LIMIT
        self.report(f"{self.ctx.reaction} WorkChain for {self.ctx.chemical_formula}: "
                    f"{self.ctx.candidates} eta for {self.ctx.reaction_path}.")

    def _construct_adsorbate_builder(self, slab, reaction, pathway):
        ML_model = settings.inputs['adsorbates']['model']
        structure = [slab]
        slab_energy = slab["energy"]
        Workflow = WorkflowFactory(ML_model.lower())
        builder = Workflow.get_builder()
        builder.input_structures = List(structure)
        if self.ctx.reaction == "OER":
            builder.local_label = Str("Adsorbates: OER on {}".format(self.ctx.chemical_formula))
        else:
            builder.local_label = Str("Adsorbates: {} on {}".format(self.ctx.reaction_path, self.ctx.chemical_formula))
        builder.code = get_code(ML_model)

        model, model_path, device = get_model_device(ML_model)
        relax_key = "adsorbates"
        job_info = {
            "job_type": "adsorbates",
            "ML_model": ML_model,
            "model_name": model,
            "model_path": model_path,
            "model_head": settings.inputs['adsorbates']['head'],
            "device": device,
            "slab_energy": slab_energy,
            "fmax": settings.inputs[relax_key]["fmax"],
            "max_steps": settings.inputs[relax_key]["max_steps"],
            "reaction": reaction,
            "pathway": pathway
        }

        builder.job_info = Dict(job_info)
        return builder

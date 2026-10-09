import os
from collections import Counter
import json
import tempfile
import yaml
from aiida.engine import WorkChain, if_
from aiida.orm import Str, Dict, SinglefileData
from aiida.plugins import CalculationFactory, WorkflowFactory
from uvsib.db.tables import DBComposition
from uvsib.db.utils import query_structure, add_slab, query_by_columns
from uvsib.workchains.utils import get_code, get_model_device
from uvsib.workflows import settings

MAX_NUM_BULK = settings.MAX_NUM_BULK
MAX_NUM_SURF = settings.MAX_NUM_SURF
_PD_VERIFICATION = settings._PD_VERIFICATION
MAX_SLABS_PER_CHUNK = 50

FaceCalculation = CalculationFactory(str(settings.inputs["face_build"]["model"]).lower())

def get_struct_uuid(chemical_formula):
    """Query structures by formula and return [structure, source, UUID] entries."""
    if _PD_VERIFICATION:
        method = "r2SCAN"
        results = query_structure({"composition": chemical_formula}, method=method) or []

        filtered_results = [
            [row.structure, row.source, str(row.structure_uuid)]
            for row in sorted(results, key=lambda row: row.energy)
            if row.source != "MPDB_ref"
       ]
    else:
        filtered_results = []
        uuid_list = []
        ml_bulk_model = settings.inputs['bulk_relax']['model']
        rows = query_by_columns(DBComposition, {"composition": chemical_formula})
        stable_structs = rows[0].stable_struct if rows else None
        if stable_structs and stable_structs.get("ml_uuid_list"):
            uuid_list = stable_structs["ml_uuid_list"]

        gate = bool(settings.inputs.get("optical_screen", {}).get("gate_surface_builder", False))
        for uuid_str in uuid_list:
            row = query_structure({"uuid": uuid_str}, method=ml_bulk_model)
            if row:
                result = row[0]
                if gate and not _light_absorbing(getattr(result, "band_info", None)):
                    continue
                filtered_results.append([result.structure, result.source, uuid_str ])
        if not filtered_results and uuid_list:
            # optical gate removed everything (or no band_info yet) -- fall back
            # to the unfiltered ML selection rather than starving the pipeline.
            for uuid_str in uuid_list:
                row = query_structure({"uuid": uuid_str}, method=ml_bulk_model)
                if row:
                    result = row[0]
                    filtered_results.append([result.structure, result.source, uuid_str])
    return filtered_results[:MAX_NUM_BULK], False

def _light_absorbing(band_info):
    """True if the no-DFT optical screen marked this bulk as absorbing visible
    light (reaction-agnostic gate). Missing / un-screened band_info -> False so
    the caller's fallback decides."""
    if not band_info:
        return False
    absorption = band_info.get("absorption") or {}
    return bool(absorption.get("absorbs_visible"))

def read_yaml(file_path):
    """Read a yaml file"""
    with open(file_path, "r", encoding="utf8") as fhandle:
        return yaml.safe_load(fhandle)

def _chunk(seq, size):
    """Split a list into consecutive chunks of at most ``size`` items."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]

def _structures_file(payload):
    """Stage ``payload`` (a JSON-serialisable list) as input_structures.json."""
    path = os.path.join(tempfile.mkdtemp(), "input_structures.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return SinglefileData(file=path)

def _model_cmdline():
    """Model/calculator CLI args for the face-build MLIP on the v100 code."""
    ml_model = settings.inputs["face_build"]["model"]
    model_name, model_path, device = get_model_device(ml_model)
    head = settings.inputs["face_build"]["head"]
    return [
        f"--ML_model={ml_model}",
        f"--model={model_name}",
        f"--model_path={model_path}",
        f"--task_name={head}",
        f"--device={device}",
    ]


def _facebuild_options():
    """Scheduler options for a slab job on the minimahopping code (generic parser).
    NOTE: the calculation ``label`` is NOT an option -- it lives at
    ``metadata.label`` (set by the callers). ``metadata.options`` is a fixed
    namespace and rejects unknown keys like 'label'.
    """
    js = settings.configs["codes"][settings.inputs["face_build"]["model"]]["job_script"]
    options = {
        "resources": {
            "num_machines": js["nodes"],
            "num_mpiprocs_per_machine": js["ntasks"],
            "num_cores_per_mpiproc": js["cpus"],
        },
        "max_wallclock_seconds": js["time"],
        "parser_name": "generic_parser",   # generic output.json -> Dict
    }
    if js.get("exclusive"):
        options["custom_scheduler_commands"] = "#SBATCH --exclusive"
    return options


class SurfaceBuilderWorkChain(WorkChain):
    """SurfaceBuilder WorkChain.

    Slab enumeration (``generate_all_slabs``, now bounded -- see
    ``codes/files/slab_generate.py`` "Robustness": symprec-ladder standardize +
    ``max_normal_search=1`` + size/symmetry skip + per-bulk walltime cap) and slab
    relaxation are two separate CalcJobs on the gnome@v100 code: one
    generation job per bulk structure, then its orthogonal slabs are split into
    balanced chunks (<= MAX_SLABS_PER_CHUNK) and relaxed in parallel. The global
    lowest-surface-energy faces (up to settings.MAX_NUM_SURF) are selected here
    after a structure's chunks all return.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("chemical_formula", valid_type=Str)

        spec.outline(
            cls.setup,
            cls.run_slabgen,
            cls.inspect_slabgen,
            cls.run_relax,
            cls.inspect_relax,
            cls.store_results,
            if_(cls.should_run_uncertainty)(
                cls.run_uncertainty,
                cls.inspect_uncertainty
            ),
            cls.final_report
        )

        spec.exit_code(300,"ERROR_CALCULATION_FAILED", message="The calculation did not finish successfully")
        spec.exit_code(301,"ERROR_NO_STRUCTURES_FOUND", message="No structures were found for the given formula")
        spec.exit_code(302,"ERROR_NO_SURFACE", message="No surface has been generated")

    def setup(self):
        """Setup and report"""
        self.ctx.chemical_formula = self.inputs.chemical_formula.value
        self.report(f"Running SurfaceBuilder WorkChain for {self.ctx.chemical_formula}")
        self.ctx.struct_uuid, self.ctx.from_manifest = \
            get_struct_uuid(self.ctx.chemical_formula)
        if not self.ctx.struct_uuid:
            self.report(f"No structures were found for {self.ctx.chemical_formula}")
            return self.exit_codes.ERROR_NO_STRUCTURES_FOUND
        source_counts = dict(Counter(source for _, source, _ in self.ctx.struct_uuid))
        self.report(f"Source of structures: {source_counts}")
        self.ctx.slabs_uuid = []  # uuid -> {"epa": float, "n_chunks": int} after the generation phase.
        self.ctx.relax_plan = {}

    def run_slabgen(self):
        """Submit one slab-generation CalcJob PER bulk structure.

        Mirrors the relax-side chunking: an SQS sweep hands over the whole
        composition grid at once, and a single all-bulks job would grow
        unpredictably long -- per-bulk jobs stay inside walltime and a
        pathological bulk fails alone instead of taking the run down."""
        for struct_dict, _, uuid_str in self.ctx.struct_uuid:
            builder = self._construct_surface_builder(uuid_str, struct_dict)
            self.to_context(**{f"slabgen_{uuid_str}": self.submit(builder)})

    def _construct_surface_builder(self, uuid_str, struct_dict):
        """Face-build (slab generation) Calculation builder for one bulk structure."""
        payload = [{"uuid": uuid_str, "structure": struct_dict,
                    "from_manifest": self.ctx.from_manifest}]
        builder = FaceCalculation.get_builder()
        builder.code = get_code(settings.inputs["face_build"]["model"])
        builder.parameters = Dict(dict={
            "job_type": "face_build",
            "cmdline_params": _model_cmdline() + [
                f"--max_miller_idx={settings.inputs['face_build']['max_miller_idx']}"],
            "staged_files": [["slab_generate.py", "aiida.py"],
                             ["_calculators.py", "_calculators.py"]],
            "retrieve_list": ["output.json"],
        })
        builder.file = {"input_structures_file": _structures_file(payload)}
        builder.metadata = {
            "options": _facebuild_options(),
            "label": f"Faces: {self.ctx.chemical_formula}",
        }
        return builder

    def _bulk_slabs(self, uuid_str):
        """This bulk's generated slabs, re-read from its own gen node (kept
        out of the workchain checkpoint on purpose -- ctx stores counts only)."""
        results = self.ctx[f"slabgen_{uuid_str}"].outputs.output_dict.get_dict().get("results", [])
        uuid2slabs = {r.get("uuid"): r.get("slabs", []) for r in results}
        return uuid2slabs.get(uuid_str, [])

    def _global_items(self):
        """The deterministic global slab list: bulks in sorted-uuid order,
        each bulk's slabs in generation order, every item carrying its bulk
        uuid + epa + global index. Attribution rides WITH the slab through
        the relax runner -- never inferred from position or job topology
        (non-converged slabs are dropped by the runner)."""
        items = []
        bulks = self.ctx.relax_plan["bulks"]
        for uuid_str in sorted(bulks):
            epa = bulks[uuid_str]["epa"]
            for enc in self._bulk_slabs(uuid_str):
                items.append({"slab": enc, "uuid": uuid_str, "epa": epa,
                              "index": len(items)})
        return items
    def inspect_slabgen(self):
        """Collect ALL generated slabs (per-bulk epa + counts), then plan the
        GLOBAL relax batching: every bulk's slabs pooled together and divided
        into chunks of <= MAX_SLABS_PER_CHUNK, instead of the old per-bulk
        chunking that submitted one under-filled job per bulk."""
        bulks = {}
        for struct_dict, source, uuid_str in self.ctx.struct_uuid:
            node = self.ctx[f"slabgen_{uuid_str}"]
            if not node.is_finished_ok:
                self.report(f"Slab generation job failed for uuid={uuid_str}; "
                            "this bulk is dropped.")
                continue
            for res in node.outputs.output_dict.get_dict().get("results", []):
                slabs = res.get("slabs", [])
                if not slabs:
                    self.report(f"Warning: no orthogonal slab generated for uuid={uuid_str}, source={source}")
                    continue
                bulks[uuid_str] = {"epa": res["epa"], "n_slabs": len(slabs)}
                self.report(f"uuid={uuid_str}, source={source}: {len(slabs)} orthogonal slabs "
                            f"(of {res.get('n_total', '?')} generated)")

        if not bulks:
            self.report("No orthogonal slabs were generated for any structure")
            return self.exit_codes.ERROR_NO_SURFACE

        # chunk composition ({uuid: count} per chunk), derived from counts in
        # the SAME sorted-uuid order _global_items uses -- lets a dead chunk
        # be reported as "which bulks lost how many slabs" without storing
        # the slabs themselves in the checkpoint
        chunk_uuids, filled = [], 0
        for uuid_str in sorted(bulks):
            left = bulks[uuid_str]["n_slabs"]
            while left:
                if filled == 0:
                    chunk_uuids.append({})
                take = min(left, MAX_SLABS_PER_CHUNK - filled)
                chunk_uuids[-1][uuid_str] = chunk_uuids[-1].get(uuid_str, 0) + take
                filled = (filled + take) % MAX_SLABS_PER_CHUNK
                left -= take

        total = sum(b["n_slabs"] for b in bulks.values())
        self.ctx.relax_plan = {"bulks": bulks, "chunk_uuids": chunk_uuids}
        self.report(f"global batching: {total} slabs from {len(bulks)} bulk(s) "
                    f"-> {len(chunk_uuids)} relax chunk(s) of <= {MAX_SLABS_PER_CHUNK}")

    def run_relax(self):
        """Submit one relaxation CalcJob per GLOBAL slab chunk."""
        items = self._global_items()
        for i, chunk in enumerate(_chunk(items, MAX_SLABS_PER_CHUNK)):
            builder = self._construct_relax_builder(chunk, i)
            self.to_context(**{f"relax_{i}": self.submit(builder)})

    def _construct_relax_builder(self, chunk, index):
        """Slab-relaxation Calculation builder for one GLOBAL slab chunk."""
        builder = FaceCalculation.get_builder()
        builder.code = get_code(settings.inputs['face_build']['model'])
        builder.parameters = Dict(dict={
            "job_type": "face_relax",
            # no --epa: it rides per slab now (chunks mix bulks)
            "cmdline_params": _model_cmdline() + [
                f"--fmax={settings.inputs['face_build']['fmax']}",
                f"--max_steps={settings.inputs['face_build']['max_steps']}"],
            "staged_files": [["slab_relax.py", "aiida.py"],
                             ["_calculators.py", "_calculators.py"]],
            "retrieve_list": ["output.json"],
        })
        builder.file = {"input_structures_file": _structures_file(chunk)}
        builder.metadata = {
            "options": _facebuild_options(),
            "label": f"SlabRelax: {self.ctx.chemical_formula} #{index}",
        }
        return builder

    def inspect_relax(self):
        """Regroup the chunk outputs BY THE ECHOED BULK UUID, then select the
        lowest-energy faces per bulk (the selection stays per-bulk and
        global-across-chunks, exactly as before)."""
        by_uuid = {uuid_str: [] for uuid_str in self.ctx.relax_plan["bulks"]}

        for i, composition in enumerate(self.ctx.relax_plan["chunk_uuids"]):
            node = self.ctx[f"relax_{i}"]
            if not node.is_finished_ok:
                losses = ", ".join(f"{u}: {n}" for u, n in composition.items())
                self.report(f"Relax chunk {i} failed; slabs lost per bulk -> {losses}")
                continue
            out = node.outputs.output_dict.get_dict()
            for rec in out.get("slabs", []):
                uuid_str = rec.get("uuid")
                if uuid_str not in by_uuid:
                    # attribution is load-bearing downstream (photocat / HSE
                    # select bulks via the slab's structure_uuid) -- a record
                    # we cannot attribute is dropped, never guessed
                    self.report(f"chunk {i}: slab record with unknown uuid "
                                f"{uuid_str!r} -- dropped")
                    continue
                by_uuid[uuid_str].append(rec)
            for fail in out.get("failed_slabs", []):
                self.report(f"chunk {i}: slab {fail.get('index')} of uuid="
                            f"{fail.get('uuid')} failed ({fail.get('reason')})")

        for uuid_str, merged in by_uuid.items():
            if not merged:
                self.report(f"Warning: no slab converged for uuid={uuid_str}")
                continue
            merged.sort(key=lambda x: x["surface_formation_energy"])
            selected = [(entry["slab"], entry["surface_formation_energy"])
                        for entry in merged[:MAX_NUM_SURF]]
            self.report(f"uuid={uuid_str}: {len(merged)} slabs converged")
            self.ctx.slabs_uuid.append([selected, uuid_str])

        if not self.ctx.slabs_uuid:
            self.report("No surface has been generated")
            return self.exit_codes.ERROR_NO_SURFACE

    def store_results(self):
        """Store results"""
        for slabs, uuid_str in self.ctx.slabs_uuid:
            for slab, surface_formation_energy in slabs:
                add_slab(uuid_str, self.ctx.chemical_formula, slab,
                         head=settings.inputs['face_build'].get('head'),
                         formation_energy=surface_formation_energy)

    def should_run_uncertainty(self):
        """Committee surface-energy uncertainty (SurfaceUncertaintyWorkChain).
        Opt-in via ``settings.SURFACE_UNCERTAINTY_ENABLED``."""
        return settings.SURFACE_UNCERTAINTY_ENABLED

    def run_uncertainty(self):
        """Submit SurfaceUncertaintyWorkChain for the stored slabs."""
        Workflow = WorkflowFactory("surfaceuncertainty")
        builder = Workflow.get_builder()
        builder.chemical_formula = Str(self.ctx.chemical_formula)
        self.to_context(**{"uncertainty": self.submit(builder)})

    def inspect_uncertainty(self):
        """Advisory stage: never fail the surface builder on the uncertainty analysis."""
        wch = self.ctx.uncertainty
        if not wch.is_finished_ok:
            self.report(f"SurfaceUncertaintyWorkChain did not finish OK (exit {wch.exit_status}); "
                        "continuing without surface-energy uncertainty.")
            return
        self.report("SurfaceUncertaintyWorkChain finished; uncertainty written for the stored slabs.")

    def final_report(self):
        """Final report"""
        self.report(f"SurfaceBuilderWorkChain for {self.ctx.chemical_formula} finished successfully")

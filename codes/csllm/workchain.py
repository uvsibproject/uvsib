import os
import json
import tempfile
from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code, Str
from aiida.plugins import CalculationFactory
from uvsib.workflows import settings


def get_options():
    """Scheduler options for the ``CSLLM`` code (see config.yaml)."""
    job_script = settings.configs["codes"]["CSLLM"]["job_script"]
    options = {
        "resources": {
            "num_machines": job_script["nodes"],
            "num_mpiprocs_per_machine": job_script["ntasks"],
            "num_cores_per_mpiproc": job_script["cpus"],
        },
        "max_wallclock_seconds": job_script["time"],
        "parser_name": "csllm_parser",
    }
    if job_script.get("exclusive"):
        options["custom_scheduler_commands"] = "#SBATCH --exclusive"
    return options


def get_cmdline(job_info):
    """Build the ``csllm.py`` CLI from the workchain ``job_info`` dict."""
    return [
        f"--weights_dir={job_info['weights_dir']}",
        f"--device={job_info['device']}",
        f"--dtype={job_info['dtype']}",
        f"--predict_method={job_info['predict_method']}",
        f"--predict_precursors={job_info['predict_precursors']}",
        f"--precursor_num_sets={job_info['precursor_num_sets']}",
        f"--max_new_tokens={job_info['max_new_tokens']}",
        f"--symprec={job_info['symprec']}",
    ]


def get_structures_file(structures):
    """Stage the ``[{"uuid", "structure"}, ...]`` list as input_structures.json."""
    if isinstance(structures, List):
        structures = structures.get_list()
    file_path = os.path.join(tempfile.mkdtemp(), "input_structures.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(structures, f)
    return SinglefileData(file=file_path)


CSLLMCalculation = CalculationFactory("csllm")


class CSLLMWorkChain(BaseRestartWorkChain):
    """Run CSLLMCalculation (synthesizability screen) with automatic restarts,
    mirroring ElectronicWorkChain."""

    _process_class = CSLLMCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("input_structures", valid_type=List)
        spec.input("code", valid_type=Code)
        spec.input("job_info", valid_type=Dict)
        spec.input("local_label", valid_type=Str)
        spec.expose_outputs(CSLLMCalculation)

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.exit_code(400, "ERROR_MAX_RESTARTS_EXCEEDED",
                       message="Maximum number of restarts exceeded for CSLLMWorkChain.")

    def setup(self):
        super().setup()
        job_info = self.inputs.job_info

        self.ctx.inputs = {
            "code": self.inputs.code,
            "file": {"input_structures_file": get_structures_file(self.inputs.input_structures)},
            "parameters": Dict(dict={
                "job_type": "csllm",
                "cmdline_params": get_cmdline(job_info),
            }),
            "metadata": {
                "options": get_options(),
                "label": "CSLLM: {}".format(self.inputs.local_label.value),
            },
        }

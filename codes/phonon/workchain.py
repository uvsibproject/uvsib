import os
import json
import tempfile

from aiida.engine import BaseRestartWorkChain, while_
from aiida.orm import List, Dict, SinglefileData, Code, Str
from aiida.plugins import CalculationFactory

from uvsib.workflows import settings


def get_options(job_info):
    """Scheduler options. Uses a dedicated ``Phonon`` codes block if present,
    else falls back to the chosen MLIP's block (phonons want longer walltime)."""
    codes = settings.configs['codes']
    job_script = codes.get('Phonon', codes[job_info['ML_model']])['job_script']
    resources = {
        'num_machines': job_script['nodes'],
        'num_mpiprocs_per_machine': job_script['ntasks'],
        'num_cores_per_mpiproc': job_script['cpus'],
    }
    options = {
        'resources': resources,
        'max_wallclock_seconds': job_script['time'],
        'parser_name': 'phonon_parser',
    }
    if job_script.get('exclusive'):
        options.update({'custom_scheduler_commands': '#SBATCH --exclusive'})
    return options


def get_structures_file(structures):
    """Stage the [{uuid, structure}, ...] request as input_structures.json.

    Accepts a plain list or an AiiDA ``List`` node (unwrapped via ``get_list``)."""
    if hasattr(structures, "get_list"):
        structures = structures.get_list()
    # own temp dir per call: chunks are staged concurrently by PhononScreenWorkChain
    path = os.path.join(tempfile.mkdtemp(), "input_structures.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(structures, f)
    return SinglefileData(file=path)


def get_cmdline(ji):
    """Build the phonon.py command line from a job_info dict."""
    cmdline = [
        f"--ML_model={ji['ML_model']}",
        f"--model={ji.get('model_name')}",
        f"--model_path={ji.get('model_path')}",
        f"--task_name={ji.get('model_head')}",
        f"--device={ji.get('device', 'cuda')}",
        f"--fmax={ji.get('fmax', 1e-3)}",
        f"--max_steps={ji.get('max_steps', 500)}",
        f"--min_supercell_length={ji.get('min_supercell_length', 8.0)}",
        f"--displacement={ji.get('displacement', 0.01)}",
        f"--mesh={ji.get('mesh', 20)}",
        f"--t_min={ji.get('t_min', 0.0)}",
        f"--t_max={ji.get('t_max', 1500.0)}",
        f"--t_step={ji.get('t_step', 50.0)}",
        f"--volume_scales={ji.get('volume_scales', '0.97,0.99,1.0,1.01,1.03')}",
        f"--imaginary_tol_THz={ji.get('imaginary_tol_THz', 0.1)}",
    ]
    return cmdline


PhononCalculation = CalculationFactory('phonon')


class PhononWorkChain(BaseRestartWorkChain):
    """BaseRestartWorkChain wrapper around PhononCalculation."""

    _process_class = PhononCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('input_structure', valid_type=List)
        spec.input("code", valid_type=Code)
        spec.input('job_info', valid_type=Dict)
        spec.input('local_label', valid_type=Str, required=False)
        spec.expose_outputs(PhononCalculation)

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.exit_code(400, 'ERROR_MAX_RESTARTS_EXCEEDED',
                       message='Maximum number of restarts exceeded for Phonon WorkChain.')

    def setup(self):
        super().setup()
        ji = self.inputs.job_info.get_dict()
        # volume_scales may arrive as a list; the runner takes a comma string
        if isinstance(ji.get('volume_scales'), (list, tuple)):
            ji['volume_scales'] = ",".join(str(x) for x in ji['volume_scales'])
        label = self.inputs.local_label.value if 'local_label' in self.inputs else 'phonon'

        self.ctx.inputs = {
            'code': self.inputs.code,
            'file': {'input_structures_file': get_structures_file(self.inputs.input_structure)},
            'parameters': Dict(dict={
                'job_type': 'phonon',
                'cmdline_params': get_cmdline(ji)}),
            'metadata': {
                'options': get_options(ji),
                'label': f'Phonon: {label}'},
        }

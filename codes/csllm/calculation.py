import os
from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common.datastructures import CalcInfo, CodeInfo
from uvsib.workflows import settings


class CSLLMCalculation(CalcJob):
    """AiiDA plugin for the CSLLM synthesizability / method / precursor screen.

    Stages ``codes/files/csllm.py`` as ``aiida.py`` and runs it
    against the ``input_structures.json`` handed in through the ``file``
    namespace (see that script's docstring for the I/O contract). The CSLLM
    weights are read in place from ``--weights_dir`` on the remote machine.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)
        spec.input_namespace("file", valid_type=SinglefileData, dynamic=True)
        spec.output("output_dict", valid_type=Dict, required=True)
        spec.exit_code(100, "ERROR_MISSING_OUTPUT", message="Required output file not found.")
        spec.exit_code(200, "ERROR_NO_RETRIEVED_FOLDER", message="The retrieved folder data node can not be accessed.")
        spec.exit_code(303, "ERROR_OUTPUT_INCOMPLETE", message="The output file is incomplete.")

    def prepare_for_submission(self, folder):
        parameters = self.inputs.parameters.get_dict()
        cmdline = parameters["cmdline_params"]

        with open(os.path.join(settings.files_path, "csllm.py"), "r", encoding="utf-8") as f:
            content = f.read()
        with folder.open("aiida.py", "w", encoding="utf-8") as f:
            f.write(content)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.retrieve_list = ["output.json"]
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (file.uuid, file.filename, file.filename) for file in self.inputs.file.values()
        ]
        calcinfo.provenance_exclude_list = ["input_structures.json"]

        return calcinfo

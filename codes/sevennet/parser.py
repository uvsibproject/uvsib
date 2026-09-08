import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict


class SevenNetParser(Parser):
    """Parser for SevenNetCalculation.

    Reads the ``output.json`` written by the shipped ``aiida.py`` runner
    (relax / face_build / adsorbates / ...) from the retrieved folder and
    exposes it as an ``output_dict`` Dict node.
    """

    def parse(self, **kwargs):
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        output_filename = 'output.json'
        if output_filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT
        with retrieved_folder.open(output_filename, 'r') as f:
            data = json.load(f)

        num_filename = 'total.txt'
        if num_filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT
        with retrieved_folder.open(num_filename, 'r') as f:
            content = f.read()
        num_structures = int(content)

        failed_filename = 'failed.txt'
        if failed_filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT
        with retrieved_folder.open(failed_filename, 'r') as f:
            content = f.read()
        num_failed = int(content)

        self.out("output_dict", Dict(dict=data))

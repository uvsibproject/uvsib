import json
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict


class GenericParser(Parser):
    """ Generic Parser output.jsob --> Dict"""
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

        self.out("output_dict", Dict(dict=data))

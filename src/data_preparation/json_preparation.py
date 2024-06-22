import json
import logging

class JSONDataLoader:
    def __init__(self, json_file, source_number):
        self.json_file = json_file
        self.source_number = source_number

    def load_source_data(self):
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        try:
            return data["sources"][self.source_number]
        except IndexError as e:
            logging.error(f"Error loading source attributes: {e}")
            raise e

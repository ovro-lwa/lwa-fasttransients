import json
import logging

class JSONDataLoader:
    def __init__(self, json_file):
        self.json_file = json_file

    def load_source_data(self):
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        try:
            return data["sources"][0] # Load the first source
        except IndexError as e:
            logging.error(f"Error loading source attributes: {e}")
            raise e

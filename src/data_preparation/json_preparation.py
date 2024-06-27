import json
import logging

class JSONDataLoader:
    """
    Class to load JSON data for the OVRO-LWA Fast Transients pipeline.

    Args:
        json_file (str): Path to the JSON file containing the metadata.
    """
    def __init__(self, json_file):
        self.json_file = json_file

    def load_source_data(self):
        """
        Load the source data from the JSON file.

        Returns:
            dict: Loaded source data.
        """
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        try:
            return data["sources"][0] # Load the first source
        except IndexError as e:
            logging.error(f"Error loading source attributes: {e}")
            raise e

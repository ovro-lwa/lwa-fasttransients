import subprocess
import logging
import os
import json
from timeit import default_timer as timer
import configparser
from datetime import datetime

class TorchDedispersionProcessor:
    def __init__(self, config_path):
        self.config_path = config_path

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(script_dir, '../config.ini')
        config = configparser.ConfigParser()
        config.read(config_file_path)
        self.pytorch_dedispersion_path = config.get('Paths', 'PyTorchDedispersionPath')

    def process(self, fil_file_name, dm_range, snr_threshold=7, boxcar_widths=[1, 2, 4, 8, 16], bad_channel_file=None):
        dedispersion_start = timer()

        logging.info(f"TORCH DEDISPERSION: Processing file {fil_file_name} using PyTorchDedispersion")

        config = {
            "SOURCE": fil_file_name,
            "SNR_THRESHOLD": snr_threshold,
            "BOXCAR_WIDTHS": boxcar_widths,
            "DM_RANGES": dm_range
        }

        if bad_channel_file:
            config["BAD_CHANNEL_FILE"] = bad_channel_file

        with open(self.config_path, 'w') as config_file:
            json.dump(config, config_file)

        # List files before running the command
        before_files = set(os.listdir('.'))

        command = [
            'python', os.path.join(self.pytorch_dedispersion_path, 'pytorch_dedispersion/dedisperse_candidates.py'),
            '--config', self.config_path,
            '--verbose', '--gpu', '0'
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing dedisperse_candidates.py: {e}")
            return None

        # List files after running the command
        after_files = set(os.listdir('.'))

        # Identify the new file
        new_files = after_files - before_files
        csv_file_name = None
        for file in new_files:
            if file.startswith('candidates_') and file.endswith('.csv'):
                csv_file_name = file
                break

        if not csv_file_name:
            logging.error(f"Expected CSV file not found after running dedisperse_candidates.py")
            return None

        dedispersion_end = timer()
        logging.debug(f"TORCH DEDISPERSION: Processing took {dedispersion_end - dedispersion_start} seconds")

        return csv_file_name

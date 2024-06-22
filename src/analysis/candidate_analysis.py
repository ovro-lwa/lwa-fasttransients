import subprocess
import logging
import pandas as pd
import glob
from timeit import default_timer as timer
import configparser
import os

class CandidateAnalyzer:
    def __init__(self):
        pass

    def analyze(self, fil_file_name):
        candcsvmaker_start = timer()

        logging.info('CANDCSVMAKER: Creating a CSV file to get all the info from all the cand files...')

        cand_files = glob.glob('*.cand')

        if not cand_files:
            logging.error("No .cand files found in the current directory.")
            return None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        candcsvmaker_path = config.get('Paths', 'CandCsvMakerScript', fallback=None)

        if not candcsvmaker_path:
            logging.error("Path to candcsvmaker.py is not defined in config.ini.")
            return None

        command = ['python', candcsvmaker_path, '-v', '-f', fil_file_name, '-k', f"{fil_file_name}_your_rfi_mask.bad_chans"] + ['-c'] + cand_files

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing candcsvmaker.py: {e}")
            return None

        csv_file_name = f"{os.path.splitext(fil_file_name)[0]}.csv"

        if not os.path.exists(csv_file_name):
            logging.error(f"Expected CSV file {csv_file_name} not found after running candcsvmaker.")
            return None

        try:
            candidates = pd.read_csv(csv_file_name)
            num_cands = str(candidates.shape[0])
        except Exception as e:
            logging.error(f"Error reading candidates CSV: {e}")
            return None

        candcsvmaker_end = timer()
        logging.debug(f'CANDMAKER: candcsvmaker.py took {candcsvmaker_end - candcsvmaker_start} s')

        return num_cands

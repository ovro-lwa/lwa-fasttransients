import subprocess
import logging
from timeit import default_timer as timer
import os
import glob
import numpy as np
import configparser 

class FileConverter:
    def __init__(self):
        pass

    #TODO: add automatic calculation of lowest frequency
    @staticmethod
    def determine_num_channels(dm, low_freq=63.2, bw=19.6):
        num_channels = np.ceil(np.sqrt(8.3 * bw**2 * (low_freq / 1000)**-3 * dm))  # smearing calculation
        num_channels = int(num_channels)
        num_channels = (num_channels + 15) // 16 * 16  # multiple of 16
        return num_channels

    def find_fits_file(self):
        current_directory = os.getcwd()
        fits_files = glob.glob(os.path.join(current_directory, '*.fits'))
        if len(fits_files) == 2:
            return fits_files[1]
        else:
            logging.error(f"Expected 2 FITS files, found {len(fits_files)}. Check the directory: {current_directory}")
            return None

    def run_conversion(self, source_data):
        file_name = source_data["file_name"]
        ra = source_data["RA"]
        dec = source_data["Dec"]
        dm = source_data["DM"]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '../config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        
        channelize_script_path = config.get('Paths', 'ChannelizeScriptPath', fallback=os.getenv('CHANNELIZE_SCRIPT_PATH'))
        hdf5_conversion_script_path = config.get('Paths', 'HDF5ConversionScriptPath', fallback=os.getenv('HDF5_CONVERSION_SCRIPT_PATH'))
        hdf_to_fil_script_path = config.get('Paths', 'HDFToFilScriptPath', fallback=os.getenv('HDF_TO_FIL_SCRIPT_PATH'))
        
        if not all([channelize_script_path, hdf5_conversion_script_path, hdf_to_fil_script_path]):
            logging.error("One or more script paths are not configured. Please check config.ini.")
            return
        
        num_channels = self.determine_num_channels(dm)
        
        start_time = timer()
        channelize_command = [
            "python", channelize_script_path, file_name,
            "-p", "-c", str(num_channels), "-r", str(ra), "-d", str(dec)
        ]
        logging.debug(f"Executing channelize command: {' '.join(channelize_command)}")
        subprocess.run(channelize_command)
        end_time = timer()
        logging.debug(f"Channelize script executed in {end_time - start_time} seconds")
        
        fits_file_name = self.find_fits_file()
        
        hdf5_start_time = timer()
        hdf5_command = ["python", hdf5_conversion_script_path, fits_file_name]
        logging.debug(f"Executing HDF5 conversion command: {' '.join(hdf5_command)}")
        subprocess.run(hdf5_command)
        hdf5_end_time = timer()
        logging.debug(f"HDF5 conversion script executed in {hdf5_end_time - hdf5_start_time} seconds")
        
        hdf_file_name = fits_file_name.replace('.fits', '.hdf5')
        hdf_to_fil_command = [
            "python", hdf_to_fil_script_path, "-n", hdf_file_name, "-RA", str(ra), "-Dec", str(dec)
        ]
        logging.debug(f"Executing HDF to FIL conversion command: {' '.join(hdf_to_fil_command)}")
        subprocess.run(hdf_to_fil_command)
        logging.debug("HDF to FIL conversion script executed.")
        
        fil_file_name = hdf_file_name.replace('.hdf5', '.fil')
        logging.info(f"Conversion step is completed. The output file is {fil_file_name}")
        
        return {
            "fits_file_name": fits_file_name,
            "hdf5_file_name": hdf_file_name,
            "fil_file_name": fil_file_name
        }

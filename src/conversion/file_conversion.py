import subprocess
import logging
from timeit import default_timer as timer
import os
import glob
import numpy as np
import configparser
import your

class FileConverter:
    """
    Class for converting voltage beam data files to filterbank files with channelization determined by the corresponding high limit on DM (hiDM in metadata).

    Methods:
        determine_num_channels(dm, low_freq, bw): Determine the number of channels needed to avoid smearing.
        find_fits_file(): Find the FITS file in the current directory. Currently finds the FITS file corresponding to the upper frequency band.
        validate_header(fil_file_name): Validate the header of the converted .fil file.
        run_conversion(source_data): Run the conversion process on source data.
    """

    # Constants for LWA higher frequency band
    # These values are specific to the LWA higher frequency band
    # and might need to be changed for different datasets.
    LOW_FREQ = 63.2  # MHz
    BANDWIDTH = 19.6  # MHz

    def __init__(self):
        pass

    @staticmethod
    def determine_num_channels(dm, low_freq=LOW_FREQ, bw=BANDWIDTH):
        """
        Determine the number of channels needed to avoid smearing.

        Args:
            dm (float): Dispersion Measure.
            low_freq (float): Lowest frequency in MHz.
            bw (float): Bandwidth in MHz.

        Returns:
            int: Number of channels.
        """
        num_channels = np.ceil(np.sqrt(8.3 * bw**2 * (low_freq / 1000)**-3 * dm))  # smearing calculation
        num_channels = int(num_channels)
        num_channels = (num_channels + 15) // 16 * 16  # multiple of 16
        return num_channels

    def find_fits_file(self):
        """
        Find the FITS file in the current directory. Currently finds the FITS file corresponding to the upper frequency band.

        Returns:
            str: Path to the FITS file, or None if not found.
        """

        current_directory = os.getcwd()
        fits_files = glob.glob(os.path.join(current_directory, '*.fits'))
        if len(fits_files) == 2:
            return fits_files[1]
        else:
            logging.error(f"Expected 2 FITS files, found {len(fits_files)}. Check the directory: {current_directory}")
            return None

    def validate_header(self, fil_file_name):
        """
        Validate the header of the converted .fil file.

        Args:
            fil_file_name (str): Name of the .fil file.

        Returns:
            None
        """
        y = your.Your(fil_file_name)
        bw = abs(y.your_header.bw)
        fch1 = y.your_header.fch1

        if not (self.LOW_FREQ <= (fch1 + bw)):
            logging.warning(f"Channelization might be wrong since {fch1 + bw} is the lowest frequency in fil file.")

        if not (abs(self.BANDWIDTH - bw) < 0.1):
            logging.warning(f"bw ({bw}) is not close to the expected BANDWIDTH ({self.BANDWIDTH})")

    def run_conversion(self, source_data):
        """
        Run the conversion process on source data.

        Args:
            source_data (dict): Metadata of the source data.

        Returns:
            dict: Filenames of the converted files.
        """
        file_name = source_data["file_name"]
        ra = source_data["RA"]
        dec = source_data["Dec"]
        dm = source_data["DM"]
        hiDM = source_data.get("hiDM", 1000)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        config_path = os.path.join(script_dir, '../config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        
        channelize_script_path = config.get('Paths', 'ChannelizeScriptPath', fallback=os.getenv('CHANNELIZE_SCRIPT_PATH'))
        hdf5_conversion_script_path = os.path.join(script_dir, 'writeHDF5FromPsrfits.py')
        
        # Updated path for the HDF to FIL conversion script
        hdf_to_fil_script_path = os.path.join(script_dir, 'hdf_to_your_fil.py')

        if not all([channelize_script_path, hdf5_conversion_script_path, hdf_to_fil_script_path]):
            logging.error("One or more script paths are not configured correctly.")
            return

        num_channels = self.determine_num_channels(hiDM)

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

        # Validate the header of the .fil file
        self.validate_header(fil_file_name)

        return {
            "fits_file_name": fits_file_name,
            "hdf5_file_name": hdf_file_name,
            "fil_file_name": fil_file_name
        }

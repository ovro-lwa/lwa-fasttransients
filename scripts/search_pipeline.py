# Parts are taken from here: https://github.com/thepetabyteproject/tpp/blob/master/tpp_pipeline.py 
# and updated for our purposes
 
import configparser
import json
import argparse
import subprocess
import logging
from timeit import default_timer as timer
import os
import glob
from your import Your
import numpy as np
import pandas as pd


class LWA_Transient_Pipeline:
    def __init__(self, json_file, source_number, checkpoint_file=None):
        self.json_file = json_file
        self.source_number = source_number
        self.checkpoint_file = checkpoint_file
        self.setup_logging()
        self.dm = None
        self.ra = None
        self.dec = None
        self.file_name = None
        self.load_attributes_from_json()
        # Initialize or load pipeline state
        self.pipeline_state = {'conversion_done': False, 'rfi_filter_done': False, 'heimdall_done': False, 'analysis_done': False, 'output_filenames': {}}
        if checkpoint_file:
            self.load_pipeline_state()

    def setup_logging(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler('pipeline.log'),
                                logging.StreamHandler()
                            ])    

    def load_attributes_from_json(self):
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        try:
            source_data = data["sources"][self.source_number]
            self.file_name = source_data["file_name"]
            self.ra = source_data["RA"]
            self.dec = source_data["Dec"]
            self.dm = source_data["DM"]
        except IndexError as e:
            logging.error(f"Error loading source attributes: {e}")

    def save_pipeline_state(self):
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as file:
                json.dump(self.pipeline_state, file)
            logging.info(f"Pipeline state saved to {self.checkpoint_file}")

    def load_pipeline_state(self):
        try:
            with open(self.checkpoint_file, 'r') as file:
                self.pipeline_state = json.load(file)
            logging.info(f"Pipeline state loaded from {self.checkpoint_file}")
        except Exception as e:
            logging.error(f"Failed to load pipeline state: {e}")


    def find_fits_file(self):
        """
        Find the FITS file in the current directory.

        Returns:
            str: The path to the FITS file.
        """
        current_directory = os.getcwd()
        fits_files = glob.glob(os.path.join(current_directory, '*.fits'))
        #TODO: Implement logic to select some predefined frequency range
        # We expect 2 FITS files, one for each Tuning. Currently take the one that contains higher frequencies.
        if len(fits_files) == 2:
            return fits_files[0]
        else:
            logging.error(f"Expected 2 FITS files, found {len(fits_files)}. Check the directory: {current_directory}")
            return None
        
    
    @staticmethod
    def determine_num_channels(dm):
        """
        Determine the number of frequency channels based on the Dispersion Measure (DM).
        
        Args:
            dm (float): The Dispersion Measure of the source.
            
        Returns:
            int: The number of frequency channels.
        """
        # TODO: Implement the logic to determine the number of frequency channels based on the DM
        num_channels = np.ceil(19.6/(np.sqrt(1/(8.3*dm*0.06379880369976164^-3)))) #smearing caluclation, to be checked
        #num_channels = 8192
        return num_channels
    
    @staticmethod
    def extract_info_from_fil(filenames, your_object, dm):
        """
        Extract information from the FIL file and check if the provided DM is less than the maximum possible for the file.

        
        Args:
            filenames (str): The path to the FIL file.
            your_object (Your): An instance of the Your class.
            dm (float): The Dispersion Measure of the source.

        Returns:
            tuple: A tuple containing the number of samples per gulp and the maximum boxcar size for heimdall searches.
        """
        logging.info("Reading raw data from "+str(filenames))

        center_freq=your_object.your_header.center_freq
        logging.info("The center frequency is "+str(center_freq)+" MHz")

        bw=your_object.your_header.bw
        logging.info("The bandwidth is "+str(bw)+" MHz")

        tsamp=your_object.your_header.native_tsamp
        logging.info("The native sampling time is "+str(tsamp)+" s")

        obs_len = your_object.your_header.native_nspectra*tsamp

        if obs_len >= 60:
                obs_len_min = obs_len/60
                logging.info("Dataset length is "+str(obs_len_min)+" minutes")
        else:
            logging.info("Dataset length is "+str(obs_len)+" seconds")

        f_low=(center_freq+bw/2)*10**(-3) #in GHz
        f_high=(center_freq-bw/2)*10**(-3) #in GHz

        logging.info("The frequency range is "+str(f_low)+" GHz to "+str(f_high)+" GHz")

        dm_h=(obs_len*10**3/4.15)*(1/((1/f_low**2)-(1/f_high**2)))

        if dm >= dm_h:
            logging.error("Invalid Dispersion Measure (DM). DM should be less than the calculated DM threshold.")

        max_delay = your_object.dispersion_delay(dms=dm)
        dispersion_delay_samples = np.ceil(max_delay / your_object.your_header.tsamp)


        if your_object.your_header.nspectra < 2**18:
            nsamps_gulp = your_object.your_header.nspectra
        else:
            nsamps_gulp = int(
                np.max([(2 ** np.ceil(np.log2(dispersion_delay_samples))), 2**18])
            )

        boxcar_max=int(50e-3 / your_object.your_header.tsamp)

        return nsamps_gulp, boxcar_max

    def do_RFI_filter(self, filenames, basenames, your_object):
        """
        Perform RFI filtering on the FIL file.

        Args:
            filenames (str): The path to the FIL file.
            basenames (str): The basename of the file.
            your_object (Your): An instance of the Your class.
        """
        mask_start=timer()
        mask_cmd="your_rfimask.py -v -f "+str(filenames)+" -sk_sigma 4 -sg_sigma 4 -sg_frequency 15"
        logging.debug('RFI MASK: command = ' + mask_cmd)
        subprocess.call(mask_cmd,shell=True)
        mask_end=timer()
        logging.debug('RFI MASK: your_rfimask.py took '+str(mask_end-mask_start)+' s')
        mask_basename=str(basenames)+'_your_rfi_mask'
        killmask_file= f"{mask_basename}.bad_chans"
        with open(killmask_file,'r') as myfile:
            file_str = myfile.read()
        
        my_list = []  # initializing a list
        for chan in file_str.split(' '):  # using split function to split the list
            my_list.append(chan)
        for chan in my_list:
            if chan == '':
                my_list.remove(chan)
        if len(my_list) == 0:
            logging.info('RFI MASK: No channels zapped')
        else:
            logging.debug(f'RFI MASK: No. of channels zapped = {len(my_list)}')
            logging.info('RFI MASK: Percentage of channels zapped = ' + str((len(my_list) / your_object.your_header.nchans) * 100) + ' %')

    @staticmethod
    def print_zap_chans_command(zap_chans):
        """
        Returns the zap channels command for Heimdall.

        Args:
            zap_chans (list): A list of channels to zap.
        """
        cmd_parts = []
        for chan in zap_chans:
            cmd_parts.append(f"-zap_chans {int(chan)} {int(chan)}")
        return " ".join(cmd_parts)


    def do_heimdall(self, filenames, killmask, dm, nsamps_gulp, boxcar_max):
        """
        Run Heimdall on the raw data using the specified parameters.

        Args:
            filenames (str): The path to the fil files.
            killmask (str): The path to the RFI mask file.
            dm (float): The expected dispersion measure value.
            nsamps_gulp (int): The number of samples to process at a time.
            boxcar_max (int): The maximum boxcar size.
        """
        heimdall_start = timer()
        logging.info(f"HEIMDALL: Using the raw data from {filenames} for Heimdall and using the RFI mask {killmask}")
        logging.info("HEIMDALL: Preparing to run Heimdall..")

        # Load bad channels mask
        try:
            mask = np.loadtxt(killmask)
            if len(mask.shape) == 1:
                bad_chans = list(mask)
                zap_chans_cmd = self.print_zap_chans_command(bad_chans)
            else:
                logging.warning("RFI mask not understood, can only be 1D. Not using RFI flagging.")
                zap_chans_cmd = ""
        except Exception as e:
            logging.error(f"Error loading RFI mask: {e}")
            zap_chans_cmd = ""

        # Load script paths from configuration file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        heimdall_binary_path = config.get('Paths', 'HeimdallBinaryPath')

        # Construct Heimdall command
        #TODO: update heimdall cmd based on dm
        heimdall_cmd = f"{heimdall_binary_path} -f {filenames} -dm {dm*0.9} {dm*1.1} -nsamps_gulp {nsamps_gulp} -boxcar_max {boxcar_max} {zap_chans_cmd}"
        logging.debug(f"Executing Heimdall command: {heimdall_cmd}")
        subprocess.call(heimdall_cmd, shell=True)
        heimdall_end = timer()
        logging.debug(f"HEIMDALL: Heimdall processing took {heimdall_end - heimdall_start} s")


    def do_candcsvmaker(self, fil_file_name, basenames, killmask):
        """
        Run the candcsvmaker.py script to create a CSV file containing information from all the cand files.

        Args:
            fil_file_name (str): The path to the fil file.
            basenames (str): The base name for the output CSV file.
            killmask (str): The path to the RFI mask file.

        Returns:
            str: The number of candidates found in the CSV file.
        """
        # Start timer
        candcsvmaker_start = timer()

        # Logging start of the process
        logging.info('CANDCSVMAKER: Creating a CSV file to get all the info from all the cand files...')

        # Find all .cand files in the current directory
        cand_files = glob.glob('*.cand')

        # Check if cand files are found
        if not cand_files:
            logging.error("No .cand files found in the current directory.")
            return None

        # Load the path to candcsvmaker.py from config.ini
        config = configparser.ConfigParser()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config.read(os.path.join(script_dir, 'config.ini'))
        candcsvmaker_path = config.get('Paths', 'CandCsvMakerScript', fallback=None)

        if not candcsvmaker_path:
            logging.error("Path to candcsvmaker.py is not defined in config.ini.")
            return None

        # Prepare the command
        command = ['python', candcsvmaker_path, '-v', '-f', fil_file_name, '-k', killmask] + ['-c'] + cand_files

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing candcsvmaker.py: {e}")
            return None

        # Construct the expected output CSV file name
        csv_file_name = f"{basenames}.csv"

        # Check if the CSV file exists after running candcsvmaker.py
        if not os.path.exists(csv_file_name):
            logging.error(f"Expected CSV file {csv_file_name} not found after running candcsvmaker.")
            return None

        # Read the generated CSV file
        try:
            candidates = pd.read_csv(csv_file_name)
            num_cands = str(candidates.shape[0])
        except Exception as e:
            logging.error(f"Error reading candidates CSV: {e}")
            return None

        # End timer and log the duration
        candcsvmaker_end = timer()
        logging.debug(f'CANDMAKER: candcsvmaker.py took {candcsvmaker_end - candcsvmaker_start} s')

        # Return the number of candidates
        return num_cands

    def run_conversion(self):
        """
        Run the conversion step of the pipeline.

        Args:
            json_file (str): Path to the JSON file containing the metadata.
            source_number (int): The index of the source in the JSON file to process.

        Returns:
            tuple: A tuple containing the path to the FIL file and the Dispersion Measure (DM).
        """

        if self.pipeline_state['conversion_done']:
            logging.info("Conversion already done, skipping to next step.")
            return self.pipeline_state['output_filenames'].get('fil_file_name')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        
        channelize_script_path = config.get('Paths', 'ChannelizeScriptPath', fallback=os.getenv('CHANNELIZE_SCRIPT_PATH'))
        hdf5_conversion_script_path = config.get('Paths', 'HDF5ConversionScriptPath', fallback=os.getenv('HDF5_CONVERSION_SCRIPT_PATH'))
        hdf_to_fil_script_path = config.get('Paths', 'HDFToFilScriptPath', fallback=os.getenv('HDF_TO_FIL_SCRIPT_PATH'))
        
        if not all([channelize_script_path, hdf5_conversion_script_path, hdf_to_fil_script_path]):
            logging.error("One or more script paths are not configured. Please check config.ini.")
            return
        
        # Utilizing attributes loaded by load_attributes_from_json()
        num_channels = self.determine_num_channels(self.dm)
        
        start_time = timer()
        channelize_command = [
            "python", channelize_script_path, self.file_name,
            "-p", "-c", str(num_channels), "-b", str(num_channels), "-r", str(self.ra), "-d", str(self.dec)
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
            "python", hdf_to_fil_script_path, "-n", hdf_file_name, "-RA", str(self.ra), "-Dec", str(self.dec)
        ]
        logging.debug(f"Executing HDF to FIL conversion command: {' '.join(hdf_to_fil_command)}")
        subprocess.run(hdf_to_fil_command)
        logging.debug("HDF to FIL conversion script executed.")
        
        fil_file_name = hdf_file_name.replace('.hdf5', '.fil')
        logging.info(f"Conversion step is completed. The output file is {fil_file_name}")
        
        self.pipeline_state['conversion_done'] = True
        self.pipeline_state['output_filenames']['fil_file_name'] = fil_file_name
        self.save_pipeline_state()
        
        return fil_file_name
    

    def run_processing(self, fil_file_name):
        """
        Run the processing step of the pipeline.

        Args:
            fil_file_name (str): The path to the FIL file.
        """

        fil_file = Your(fil_file_name)
        nsamps_gulp, boxcar_max = self.extract_info_from_fil(fil_file_name, fil_file, self.dm)
        
        # Check if RFI Filtering is already done
        if not self.pipeline_state['rfi_filter_done']:
            logging.info("Starting RFI Filtering...")
            self.do_RFI_filter(fil_file_name, fil_file.your_header.basename, fil_file)
            self.pipeline_state['rfi_filter_done'] = True
            self.save_pipeline_state()
        else:
            logging.info("RFI Filtering already completed, skipping.")

        # Check if Heimdall Processing is already done
        if not self.pipeline_state['heimdall_done']:
            logging.info("Starting Heimdall Processing...")
            killmask = f"{fil_file.your_header.basename}_your_rfi_mask.bad_chans"
            self.do_heimdall(fil_file_name, killmask, self.dm, nsamps_gulp, boxcar_max)
            self.pipeline_state['heimdall_done'] = True
            self.save_pipeline_state()
        else:
            logging.info("Heimdall Processing already completed, skipping.")

        return 

    def run_analysis(self, fil_file_name):
        """
        Run the analysis step of the pipeline.
        
        Args:
            fil_file_name (str): The path to the FIL file.
        """
        if not self.pipeline_state['analysis_done']:
            logging.info("Starting analysis...")
            # Assuming basename can be derived from fil_file_name
            basenames = os.path.splitext(fil_file_name)[0]
            killmask = f"{basenames}_your_rfi_mask.bad_chans"
            num_cands = self.do_candcsvmaker(fil_file_name, basenames, killmask)
            logging.info(f"Analysis completed. Number of candidates: {num_cands}")
            self.pipeline_state['analysis_done'] = True
            self.save_pipeline_state()
        else:
            logging.info("Analysis already completed, skipping.")

    def run(self):
        fil_file_name = self.run_conversion()
        if fil_file_name:
            self.run_processing(fil_file_name)
            self.run_analysis(fil_file_name)
        else:
            logging.error("Conversion step failed, processing cannot continue.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific source from the JSON metadata.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the metadata.')
    parser.add_argument('source_number', type=int, help='The index of the source in the JSON file to process.')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file to resume pipeline.', default=None)
    
    args = parser.parse_args()
    pipeline = LWA_Transient_Pipeline(args.json_file, args.source_number, args.checkpoint)
    pipeline.run()

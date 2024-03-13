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

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('pipeline.log'),  # Writes log to a file
                        logging.StreamHandler()  # Prints log to stdout
                    ])



def find_fits_file():
    """
    Find the FITS file in the current directory.

    Returns:
        str: The path to the FITS file.
    """
    # Get the current directory
    current_directory = os.getcwd()
    # List all FITS files in the current directory
    fits_files = glob.glob(os.path.join(current_directory, '*.fits'))
    
    # Assuming there's only two FITS file as per your setup
    if len(fits_files) == 2:
        return fits_files[0]
    else:
        # Handle the case where no or multiple FITS files are found
        logging.error(f"Expected 2 FITS files, found {len(fits_files)}. Check the directory: {current_directory}")
        return None
    

def determine_num_channels(dm):
    """
    Determine the number of frequency channels based on the Dispersion Measure (DM).
    
    Args:
        dm (float): The Dispersion Measure of the source.
        
    Returns:
        int: The number of frequency channels.
    """
    # TODO: Implement the logic to determine the number of frequency channels based on the DM
    num_channels = 8192
    return num_channels

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


def do_RFI_filter(filenames,basenames,your_object):
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


def do_heimdall(filenames, killmask, dm, nsamps_gulp, boxcar_max):
    heimdall_start = timer()
    logging.info(f"HEIMDALL: Using the raw data from {filenames} for Heimdall and using the RFI mask {killmask}")
    logging.info("HEIMDALL: Preparing to run Heimdall..")

    # Load bad channels mask
    try:
        mask = np.loadtxt(killmask)
        if len(mask.shape) == 1:
            bad_chans = list(mask)
            zap_chans_cmd = print_zap_chans_command(bad_chans)
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
    heimdall_cmd = f"{heimdall_binary_path} -f {filenames} -dm 0 {dm*1.1} -nsamps_gulp {nsamps_gulp} -boxcar_max {boxcar_max} {zap_chans_cmd}"
    logging.debug(f"Executing Heimdall command: {heimdall_cmd}")
    subprocess.call(heimdall_cmd, shell=True)
    heimdall_end = timer()
    logging.debug(f"HEIMDALL: Heimdall processing took {heimdall_end - heimdall_start} s")





    

def run_conversion(json_file, source_number):
    """
    Run the conversion step of the pipeline.

    Args:
        json_file (str): Path to the JSON file containing the metadata.
        source_number (int): The index of the source in the JSON file to process.

    Returns:
        tuple: A tuple containing the path to the FIL file and the Dispersion Measure (DM).
    """
    # Load script paths from configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to config.ini, which is in the same directory as the script
    config_path = os.path.join(script_dir, 'config.ini')

    # Load script paths from configuration file
    config = configparser.ConfigParser()
    config.read(config_path)
    channelize_script_path = config.get('Paths', 'ChannelizeScriptPath', fallback=os.getenv('CHANNELIZE_SCRIPT_PATH'))
    hdf5_conversion_script_path = config.get('Paths', 'HDF5ConversionScriptPath', fallback=os.getenv('HDF5_CONVERSION_SCRIPT_PATH'))
    hdf_to_fil_script_path = config.get('Paths', 'HDFToFilScriptPath', fallback=os.getenv('HDF_TO_FIL_SCRIPT_PATH'))

    if not all([channelize_script_path, hdf5_conversion_script_path, hdf_to_fil_script_path]):
        logging.error("One or more script paths are not configured. Please check config.ini.")
        return

    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    try:
        source_data = data["sources"][source_number]
    except IndexError:
        logging.error(f"Source number {source_number} is out of range.")
        return

    file_name = source_data["file_name"]
    ra = source_data["RA"]
    dec = source_data["Dec"]
    dm = source_data["DM"]

    # Determine the number of frequency channels
    num_channels = determine_num_channels(dm)

    # First script execution (volage beam to FITS file)
    start_time = timer()
    channelize_command = [
        "python", channelize_script_path, file_name,
        "-p", "-c", str(num_channels), "-b", str(num_channels), "-r", str(ra), "-d", str(dec)  # Example values
    ]
    logging.debug(f"Executing channelize command: {' '.join(channelize_command)}")
    subprocess.run(channelize_command)
    end_time = timer()
    logging.debug(f"Channelize script executed in {end_time - start_time} seconds")

    # Find the FITS file in the current directory
    fits_file_name = find_fits_file()

    # Second script execution (fits to hdf5)
    hdf5_start_time = timer()
    hdf5_command = [
        "python", hdf5_conversion_script_path, fits_file_name
    ]
    logging.debug(f"Executing HDF5 conversion command: {' '.join(hdf5_command)}")
    subprocess.run(hdf5_command)
    hdf5_end_time = timer()
    logging.debug(f"HDF5 conversion script executed in {hdf5_end_time - hdf5_start_time} seconds")

    hdf_file_name = fits_file_name.replace('.fits', '.hdf5')
    # Execute the third script (HDF5 to FIL)
    hdf_to_fil_command = [
        "python", hdf_to_fil_script_path,
        "-n", hdf_file_name,  # Use the HDF file name here
        "-RA", str(ra),
        "-Dec", str(dec)
    ]
    logging.debug(f"Executing HDF to FIL conversion command: {' '.join(hdf_to_fil_command)}")
    subprocess.run(hdf_to_fil_command)
    logging.debug("HDF to FIL conversion script executed.")

    fil_file_name = hdf_file_name.replace('.hdf5', '.fil')
    logging.info(f"Conversion step is completed. The output file is {fil_file_name}")

    return fil_file_name, dm


def run_processing(fil_file_name, dm):
    """
    Run the processing step of the pipeline.

    Args:
        fil_file_name (str): The path to the FIL file.
        dm (float): The Dispersion Measure of the source.
    """
    fil_file = Your(fil_file_name)
    nsamps_gulp, boxcar_max = extract_info_from_fil(fil_file_name, fil_file, dm)
    do_RFI_filter(fil_file_name, fil_file.your_header.basename, fil_file)
    killmask = f"{fil_file.your_header.basename}_your_rfi_mask.bad_chans"
    do_heimdall(fil_file_name, killmask, dm, nsamps_gulp, boxcar_max)

    # TODO: Add the rest of the processing steps here
    


def run_pipeline(json_file, source_number):
    """
    Run the pipeline for a specific source from the JSON metadata.

    Args:
        json_file (str): Path to the JSON file containing the metadata.
        source_number (int): The index of the source in the JSON file to process.
    """
    fil_file_name, dm = run_conversion(json_file, source_number)
    run_processing(fil_file_name, dm)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific source from the JSON metadata.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the metadata.')
    parser.add_argument('source_number', type=int, help='The index of the source in the JSON file to process.')
    
    args = parser.parse_args()
    run_pipeline(args.json_file, args.source_number)

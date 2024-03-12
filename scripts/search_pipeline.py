import configparser
import json
import argparse
import subprocess
import logging
from timeit import default_timer as timer
import os
import glob


# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='pipeline.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')



def find_fits_file():
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
    


def run_pipeline(json_file, source_number):
    # Load script paths from configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')
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

    # First script execution
    start_time = timer()
    channelize_command = [
        "python", channelize_script_path, file_name,
        "-p", "-c", "8192", "-b", "8192", "-r", ra, "-d", dec  # Example values
    ]
    logging.debug(f"Executing channelize command: {' '.join(channelize_command)}")
    subprocess.run(channelize_command)
    end_time = timer()
    logging.debug(f"Channelize script executed in {end_time - start_time} seconds")

    # Find the FITS file in the current directory
    fits_file_name = find_fits_file()

    # Second script execution
    hdf5_start_time = timer()
    hdf5_command = [
        "python", hdf5_conversion_script_path, fits_file_name
    ]
    logging.debug(f"Executing HDF5 conversion command: {' '.join(hdf5_command)}")
    subprocess.run(hdf5_command)
    hdf5_end_time = timer()
    logging.debug(f"HDF5 conversion script executed in {hdf5_end_time - hdf5_start_time} seconds")

    hdf_file_name = fits_file_name.replace('.fits', '.hdf')
    # Execute the third script (HDF to FIL)
    hdf_to_fil_command = [
        "python", hdf_to_fil_script_path,
        "--name", hdf_file_name,  # Use the HDF file name here
        "--RA", str(ra),
        "--Dec", str(dec)
    ]
    logging.debug(f"Executing HDF to FIL conversion command: {' '.join(hdf_to_fil_command)}")
    subprocess.run(hdf_to_fil_command)
    logging.debug("HDF to FIL conversion script executed.")

    fil_file_name = hdf_file_name+'.fil'
    logging.info(f"Conversion step is completed. The output file is {fil_file_name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific source from the JSON metadata.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the metadata.')
    parser.add_argument('source_number', type=int, help='The index of the source in the JSON file to process.')
    
    args = parser.parse_args()
    run_pipeline(args.json_file, args.source_number)

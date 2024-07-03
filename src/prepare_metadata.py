import argparse
import json
import os

def prepare_metadata(file_name, ra, dec, dm, lo_dm, hi_dm, json_file):
    """
    Prepares and saves source metadata (see args) to a JSON file.

    Args:
        file_name (str): The name of the voltage beam file.
        ra (float): The Right Ascension of the source in deg.
        dec (float): The Declination of the source in deg.
        dm (float): The Dispersion Measure of the source (in pc/cm³).
        lo_dm (float): The lower bound of the DM range (in pc/cm³).
        hi_dm (float): The upper bound of the DM range (in pc/cm³).
        json_file (str): The path to the JSON file to save the metadata to.

    If the specified JSON file does not exist, this function creates a new one and adds the metadata to it.
    """
    if not (lo_dm <= dm <= hi_dm):
        raise ValueError(f"DM ({dm}) must be within the range [{lo_dm}, {hi_dm}]")

    data = {
        "file_name": file_name,
        "RA": ra,
        "Dec": dec,
        "DM": dm,
        "loDM": lo_dm,
        "hiDM": hi_dm
    }
    
    # Check if the JSON file already exists
    if os.path.exists(json_file):
        # Read the existing data
        with open(json_file, 'r') as file:
            existing_data = json.load(file)
        existing_data['sources'].append(data)
    else:
        # Create new data structure if the file doesn't exist
        existing_data = {"sources": [data]}
    
    # Write the updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(existing_data, file, indent=4)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Prepares and saves source metadata to a JSON file used to run the pipeline.')
    parser.add_argument('--file_name', '-f', required=True, help='The name of the voltage beam file.')
    parser.add_argument('--ra', type=float, required=True, help='The Right Ascension of the source (in deg).')
    parser.add_argument('--dec', type=float, required=True, help='The Declination of the source (in deg).')
    parser.add_argument('--dm', type=float, required=True, help='The Dispersion Measure of the source.')
    parser.add_argument('--lo_dm', type=float, required=True, help='The lower bound of the DM range (in pc/cm³).')
    parser.add_argument('--hi_dm', type=float, required=True, help='The upper bound of the DM range (in pc/cm³).')
    parser.add_argument('--json_file', '-json_f', required=True, help='The JSON file to save the data to.')

    # Parse the arguments
    args = parser.parse_args()

    # Prepare and save the data
    prepare_metadata(args.file_name, args.ra, args.dec, args.dm, args.lo_dm, args.hi_dm, args.json_file)

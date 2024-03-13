import argparse
import json
import os

def prepare_data(file_name, ra, dec, dm, json_file):
    """
    Prepares and saves source data to a JSON file.

    Args:
        file_name (str): The name of the file.
        ra (float): The Right Ascension of the source.
        dec (float): The Declination of the source.
        dm (float): The Dispersion Measure of the source.
        json_file (str): The path to the JSON file to save the data to.

    If the specified JSON file does not exist, this function creates a new one and adds the data to it.
    If the file exists, it appends the new data to the existing file.
    """
    data = {
        "file_name": file_name,
        "RA": ra,
        "Dec": dec,
        "DM": dm
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
    parser = argparse.ArgumentParser(description='Prepare data for astronomical sources and save to a JSON file.')
    parser.add_argument('--file_name', '-f', required=True, help='The name of the voltage beam file.')
    parser.add_argument('--ra', type=float, required=True, help='The Right Ascension of the source (in deg).')
    parser.add_argument('--dec', type=float, required=True, help='The Declination of the source (in deg).')
    parser.add_argument('--dm', type=float, required=True, help='The Dispersion Measure of the source.')
    parser.add_argument('--json_file', '-json_f', required=True, help='The JSON file to save the data to.')

    # Parse the arguments
    args = parser.parse_args()

    # Prepare and save the data
    prepare_data(args.file_name, args.ra, args.dec, args.dm, args.json_file)

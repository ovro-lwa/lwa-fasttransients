# Pipeline description

## Overview

This repository contains tools for preparing sources data in JSON format and running a search pipeline to process this data. The pipeline includes data preparation, conversion, RFI filtering, and running the Heimdall software for transient detection. This README provides instructions on how to prepare data and run the pipeline.



## Preparing Your Data

### Step 1: Prepare JSON Data

The first step is to prepare the source data in JSON format. Use the `prepare_data.py` script to add new source data to a JSON file. If the JSON file does not exist, the script will create it.

#### Usage:

```
python prepare_data.py --file_name <file_name> --ra <RA> --dec <DEC> --dm <DM> --json_file <path_to_json_file>
```

Arguments:
- `--file_name (-f)`: The name of the voltage beam file.
- `--ra`: The Right Ascension of the source (in degrees).
- `--dec`: The Declination of the source (in degrees).
- `--dm`: The expected Dispersion Measure of the source.
- `--json_file (-json_f)`: The JSON file to save the data to.

### Step 2: Run the Search Pipeline

After preparing your JSON data file, you can run the search pipeline to process the data.

#### Usage:


```bash
python search_pipeline.py <path_to_json_file> <source_number> --checkpoint <path_to_checkpoint_file>
```

Arguments:
- `<path_to_json_file>`: Path to the JSON file containing the metadata.
- `<source_number>`: The index of the source in the JSON file to process.
- `--checkpoint`: (Optional) Path to the checkpoint file. If this file exists, the pipeline resumes from the last saved state. If it does not exist, the pipeline starts from the beginning, and the state will be saved to this file for future resumptions.


## Pipeline Steps

1. **Conversion**: Converts voltage beam files to `.fil` format, passing through FITS and HDF5 intermediary formats. It involves channelization, FITS to HDF5 conversion, and HDF5 to FIL conversion.

2. **RFI Filtering**: Applies RFI filtering on the `.fil` file, creating a bad channels mask.

3. **Heimdall Processing**: Runs the Heimdall software for transient detection on the RFI-filtered `.fil` file.

4. **Candidate Analysis**: Analyzes the output from Heimdall to aggregate candidate detections into a summary CSV file.



## Configuration

Make sure to make `config.ini` with the paths to the necessary scripts and binaries (e.g., Heimdall binary path) before running the pipeline (see config.ini.template).

## Logs

The pipeline logs its progress and any errors encountered. Check `pipeline.log` for detailed information about the execution of each step.

## Troubleshooting

- Ensure all dependencies are correctly installed and configured.
- Verify that the `config.ini` file contains correct paths to scripts and binaries.
- Check the log file for any errors that may have occurred during execution.




# OVRO-LWA Fast Transients

This repository contains the code for the OVRO-LWA Fast Transients project.

## Description

The project is dedicated to building pipelines that aim to detect and analyze fast transients with the OVRO-LWA. 

## Installation

To run the pipeline, the following software needs to be installed:
- [LWA Software Library](https://github.com/lwa-project/lsl)
- [PSRFITS utils](https://github.com/lwa-project/psrfits_utils)
- [LWA1 Pulsar Scripts](https://github.com/lwa-project/pulsar)
- [Your Unified Reader](https://github.com/thepetabyteproject/your)
- [High dimensional Interactive Plotting](https://github.com/facebookresearch/hiplot)
- [PyTorch Dedispersion](https://github.com/nkosogor/PyTorchDedispersion)



## Pipeline Overview

This repository contains tools for preparing source data in JSON format and running a search pipeline to process this data. The pipeline includes data preparation, conversion, RFI filtering, dedispersion, and visualization using HiPlot.

## Preparing Your Data

### Step 1: Prepare JSON Data

The first step is to prepare the source data in JSON format. Use the `prepare_metadata.py` script to add new source data to a JSON file. If the JSON file does not exist, the script will create it.

#### Usage:

```bash
python src/prepare_metadata.py --file_name <file_name> --ra <RA> --dec <DEC> --dm <DM> --lo_dm <loDM> --hi_dm <hiDM> --json_file <path_to_json_file>
```

Arguments:
- `--file_name (-f)`: The name of the voltage beam file.
- `--ra`: The Right Ascension of the source (in degrees).
- `--dec`: The Declination of the source (in degrees).
- `--dm`: The expected Dispersion Measure of the source (in pc/cm³).
- `--lo_dm`: The lower bound of the DM range (in pc/cm³).
- `--hi_dm`: The upper bound of the DM range (in pc/cm³).
- `--json_file (-json_f)`: The JSON file to save the data to.

### Step 2: Run the Search Pipeline

After preparing your JSON data file, you can run the search pipeline to process the data.

#### Usage:

```bash
python src/pipeline.py <path_to_json_file> <path_to_checkpoint_file>
```

Arguments:
- `<path_to_json_file>`: Path to the JSON file containing the metadata.
- `<path_to_checkpoint_file>`: Path to the checkpoint file to resume the pipeline.

## Pipeline Steps

1. **Conversion**: Converts voltage beam files to .fil format with appropriate channelization to minimize DM smearing effects.

2. **RFI Filtering**: Applies RFI filtering on the `.fil` file, creating a bad channels mask.

3. **Dedispersion**: Processes the `.fil` file to generate dedispersed data and save it as a CSV file containing candidate detections.

4. **Visualization**: Uses HiPlot to visualize the candidate detections and save the visualization as an HTML file.

## Configuration

Make sure to create a `config.ini` file with the paths to the necessary scripts and binaries before running the pipeline (see `config.ini.template`).

## Logs

The pipeline logs its progress and any errors encountered. Check `pipeline.log` for detailed information about the execution of each step.

## License

This project is licensed under the [MIT License](LICENSE).
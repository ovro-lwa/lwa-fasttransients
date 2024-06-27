# OVRO-LWA Fast Transients

This repository contains the code for the OVRO-LWA Fast Transients project.

## Description

The project is dedicated to building pipelines that aim to detect and analyze fast transients with the OVRO-LWA. 

## Quick Start

### Installation

To run the pipeline, the following software needs to be installed:
- [LWA Software Library](https://github.com/lwa-project/lsl)
- [PSRFITS utils](https://github.com/lwa-project/psrfits_utils)
- [LWA1 Pulsar Scripts](https://github.com/lwa-project/pulsar)
- [Your Unified Reader](https://github.com/thepetabyteproject/your)
- [High dimensional Interactive Plotting](https://github.com/facebookresearch/hiplot)
- [PyTorch Dedispersion](https://github.com/nkosogor/PyTorchDedispersion)

### Basic Usage

1. **Prepare JSON Data**

    python src/prepare_metadata.py --file_name <file_name> --ra <RA> --dec <DEC> --dm <DM> --lo_dm <loDM> --hi_dm <hiDM> --json_file <path_to_json_file>

2. **Run the Search Pipeline**

    python src/pipeline.py <path_to_json_file> <path_to_checkpoint_file>

For detailed instructions, visit the [Wiki](https://github.com/ovro-lwa/lwa-fasttransients/wiki).

## Key Features

- **Conversion**: Converts voltage beam files to .fil format.
- **RFI Filtering**: Applies RFI filtering on the `.fil` file.
- **Dedispersion**: Processes the `.fil` file to generate dedispersed data.
- **Visualization**: Uses HiPlot to visualize the candidate detections.

## Configuration

Create a `config.ini` file with the necessary paths (see `config.ini.template`).

## Links

- https://github.com/ovro-lwa/lwa-fasttransients/wiki - Detailed documentation, troubleshooting, and advanced usage.

## Logs

Check `pipeline.log` for detailed information about the execution of each step.

## License

This project is licensed under the MIT License (LICENSE).




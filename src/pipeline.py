import argparse
import logging
from data_preparation.json_preparation import JSONDataLoader
from conversion.file_conversion import FileConverter
from rfi_filtering.rfi_filter import RFI_Filter
from transient_detection.torch_dedispersion import TorchDedispersionProcessor
from utils.logging_setup import setup_logging
from visualization.hiplot_visualizer import HiPlotVisualizer
from dm_range_calculation.dm_range_calculator import calculate_dm_ranges
import json
import os
from your import Your
import pandas as pd

class LWA_Transient_Pipeline:
    def __init__(self, json_file, checkpoint_file, remove_trend=False, window_size=20000):
        self.json_loader = JSONDataLoader(json_file)
        self.file_converter = FileConverter()
        self.rfi_filter = RFI_Filter()
        self.dedispersion_processor = TorchDedispersionProcessor('config.json')
        self.hiplot_visualizer = HiPlotVisualizer(output_dir='.')  # Output directory for HiPlot files
        self.checkpoint_file = checkpoint_file
        self.remove_trend = remove_trend
        self.window_size = window_size
        self.setup_pipeline_state()
        setup_logging()

    def setup_pipeline_state(self):
        self.pipeline_state = {
            'conversion_done': False,
            'rfi_filter_done': False,
            'dedispersion_done': False,
            'hiplot_done': False,
            'output_filenames': {},
            'bad_channels_file': None,
            'candidate_file': None,
            'hiplot_file': None
        }
        if self.checkpoint_file and os.path.exists(self.checkpoint_file):
            self.load_pipeline_state()
        else:
            self.save_pipeline_state()

    def load_pipeline_state(self):
        try:
            with open(self.checkpoint_file, 'r') as file:
                self.pipeline_state = json.load(file)
        except Exception as e:
            logging.error(f"Failed to load pipeline state: {e}")

    def save_pipeline_state(self):
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as file:
                json.dump(self.pipeline_state, file)

    def calculate_dm_ranges(self, loDM, hiDM, fil_file_name):
        your_object = Your(fil_file_name)
        dt = your_object.your_header.tsamp
        f_ctr = your_object.your_header.center_freq
        BW = abs(your_object.your_header.bw)
        numchan = your_object.your_header.nchans

        dm_ranges = calculate_dm_ranges(loDM, hiDM, f_ctr, BW, numchan, dt)
        return dm_ranges

    def run(self):
        source_data = self.json_loader.load_source_data()  # Load the single source

        if not self.pipeline_state['conversion_done']:
            file_names = self.file_converter.run_conversion(source_data)
            self.pipeline_state['conversion_done'] = True
            self.pipeline_state['output_filenames'] = file_names
            self.save_pipeline_state()
        else:
            file_names = self.pipeline_state['output_filenames']

        fil_file_name = file_names.get('fil_file_name')
        if not fil_file_name:
            logging.error("fil_file_name not found in the output filenames.")
            return

        if not self.pipeline_state['rfi_filter_done']:
            your_object = Your(fil_file_name)
            self.rfi_filter.apply_filter(fil_file_name, os.path.splitext(fil_file_name)[0], your_object)
            bad_channels_file = f"{os.path.splitext(fil_file_name)[0]}_your_rfi_mask.bad_chans"
            self.pipeline_state['rfi_filter_done'] = True
            self.pipeline_state['bad_channels_file'] = bad_channels_file
            self.save_pipeline_state()
        else:
            bad_channels_file = self.pipeline_state['bad_channels_file']

        if not self.pipeline_state['dedispersion_done']:
            loDM = source_data.get('loDM', 50)
            hiDM = source_data.get('hiDM', 1000)
            dm_ranges = self.calculate_dm_ranges(loDM, hiDM, fil_file_name)

            csv_file_name = self.dedispersion_processor.process(
                fil_file_name, 
                dm_ranges, 
                bad_channel_file=bad_channels_file,
                remove_trend=self.remove_trend,
                window_size=self.window_size
            )
            if csv_file_name:
                self.pipeline_state['dedispersion_done'] = True
                self.pipeline_state['output_filenames']['csv_file_name'] = csv_file_name
                self.pipeline_state['candidate_file'] = csv_file_name
                self.save_pipeline_state()
            else:
                logging.error("Dedispersion processing failed. Exiting pipeline.")
                return

        if not self.pipeline_state['hiplot_done']:
            hiplot_file = self.hiplot_visualizer.generate_hiplot(self.pipeline_state['candidate_file'])
            self.pipeline_state['hiplot_done'] = True
            self.pipeline_state['hiplot_file'] = hiplot_file
            self.save_pipeline_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific source from the JSON metadata.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the metadata.')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file to resume pipeline.')
    parser.add_argument('--remove_trend', action='store_true', help='Enable trend removal')
    parser.add_argument('--window_size', type=int, default=20000, help='Window size for trend removal')

    args = parser.parse_args()
    pipeline = LWA_Transient_Pipeline(args.json_file, args.checkpoint, args.remove_trend, args.window_size)
    pipeline.run()

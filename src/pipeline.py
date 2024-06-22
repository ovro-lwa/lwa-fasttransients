import argparse
import logging
from data_preparation.json_preparation import JSONDataLoader
from conversion.file_conversion import FileConverter
from rfi_filtering.rfi_filter import RFI_Filter
from transient_detection.heimdall_processing import HeimdallProcessor
from analysis.candidate_analysis import CandidateAnalyzer
from utils.logging_setup import setup_logging
import json
import os
from your import Your

class LWA_Transient_Pipeline:
    def __init__(self, json_file, source_number, checkpoint_file=None):
        self.json_loader = JSONDataLoader(json_file, source_number)
        self.file_converter = FileConverter()
        self.rfi_filter = RFI_Filter()
        self.heimdall_processor = HeimdallProcessor()
        self.candidate_analyzer = CandidateAnalyzer()
        self.checkpoint_file = checkpoint_file
        self.setup_pipeline_state()
        setup_logging()

    def setup_pipeline_state(self):
        self.pipeline_state = {'conversion_done': False, 'rfi_filter_done': False, 'heimdall_done': False, 'analysis_done': False, 'output_filenames': {}}
        if self.checkpoint_file:
            self.load_pipeline_state()

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

    def run(self):
        source_data = self.json_loader.load_source_data()
        if not self.pipeline_state['conversion_done']:
            fil_file_name = self.file_converter.run_conversion(source_data)
            self.pipeline_state['conversion_done'] = True
            self.pipeline_state['output_filenames']['fil_file_name'] = fil_file_name
            self.save_pipeline_state()
        else:
            fil_file_name = self.pipeline_state['output_filenames']['fil_file_name']

        if not self.pipeline_state['rfi_filter_done']:
            self.rfi_filter.apply_filter(fil_file_name, os.path.splitext(fil_file_name)[0], Your(fil_file_name))
            self.pipeline_state['rfi_filter_done'] = True
            self.save_pipeline_state()

        if not self.pipeline_state['heimdall_done']:
            self.heimdall_processor.process(fil_file_name, source_data['DM'])
            self.pipeline_state['heimdall_done'] = True
            self.save_pipeline_state()

        if not self.pipeline_state['analysis_done']:
            num_cands = self.candidate_analyzer.analyze(fil_file_name)
            logging.info(f"Analysis completed. Number of candidates: {num_cands}")
            self.pipeline_state['analysis_done'] = True
            self.save_pipeline_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the pipeline for a specific source from the JSON metadata.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing the metadata.')
    parser.add_argument('source_number', type=int, help='The index of the source in the JSON file to process.')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file to resume pipeline.', default=None)

    args = parser.parse_args()
    pipeline = LWA_Transient_Pipeline(args.json_file, args.source_number, args.checkpoint)
    pipeline.run()

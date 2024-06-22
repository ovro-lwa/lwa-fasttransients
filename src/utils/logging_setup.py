import logging


def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler('pipeline.log'),
                            logging.StreamHandler()
                        ])

import logging


def setup_logging():
    """
    Set up logging for the pipeline.
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler('pipeline.log'),
                            logging.StreamHandler()
                        ])

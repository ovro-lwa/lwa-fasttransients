import configparser

class ConfigLoader:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_config(self):
        return self.config

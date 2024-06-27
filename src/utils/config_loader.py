import configparser

class ConfigLoader:
    """
    Class to load configuration settings.

    Args:
        config_path (str): Path to the configuration file.
    """
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_config(self):
        """
        Get the loaded configuration settings.

        Returns:
            configparser.ConfigParser: The loaded configuration settings.
        """
        return self.config

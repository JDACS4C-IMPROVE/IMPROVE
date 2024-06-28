import os
import sys
import logging
from pathlib import Path
from improvelib.initializer.config import Config
from improvelib.initializer.cli_params_def import improve_preprocess_conf, improve_basic_conf

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)


class PreprocessConfig(Config):
    """Class to handle configuration files for Preprocessing."""

    # Set section for config file
    section = 'Preprocess'

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("Preprocess")
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.INFO))

        # check usage of self.options
        self.options = improve_basic_conf + improve_preprocess_conf

        # Add options for Benchmark Data Format
        p = self.cli.parser

    def get_param(self, key):
        """Get a parameter from the Preprocessing config."""
        return super().get_param(PreprocessConfig.section, key)

    def set_params(self, key=None, value=None):
        print("set_params" + type(self))
        return super().set_param(PreprocessConfig.section, key, value)

    def set_param(self, key=None, value=None):
        return super().set_param(PreprocessConfig.section, key, value)

    def dict(self):
        """Get the Preprocessing config as a dictionary."""
        return super().dict(PreprocessConfig.section)

    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        """Initialize Command line Interfcace and config for Preprocessing."""
        self.logger.debug("Initializing parameters for Preprocessing.")
        # print( "initialize_parameters" + str(type(self)) )

        if additional_definitions:
            if isinstance(additional_definitions, str) or isinstance(additional_definitions, Path):
                additional_definitions = self.load_parameters(
                    additional_definitions)
            else:
                self.options = self.options + additional_definitions

        p = super().initialize_parameters(pathToModelDir, section,
                                          default_config, default_model, self.options, required)
        print(self.get_param("log_level"))
        self.logger.setLevel(self.get_param("log_level"))
        return p


if __name__ == "__main__":
    p = PreprocessConfig()
    p.initialize_parameters(pathToModelDir=".")
    print(p.dict())

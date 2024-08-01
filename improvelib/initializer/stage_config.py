import os
import sys
import logging
from pathlib import Path

from improvelib.initializer.config import Config
from improvelib.initializer.cli_params_def import (
    improve_basic_conf,
    improve_preprocess_conf,
    improve_train_conf,
    improve_infer_conf,
)

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)


class SectionConfig(Config):

    def __init__(self, section, stage_config_parameters) -> None:
        super().__init__()
        self.section = section
        self.logger = logging.getLogger(self.section)
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.INFO))

        # check usage of self.options
        self.options = []
        self.cli.set_command_line_options(
            improve_basic_conf, 'IMPROVE options')
        self.cli.set_command_line_options(
            options=stage_config_parameters, group=f'{self.section} stage options')

        # Add options for Benchmark Data Format
        p = self.cli.parser

    """
    def get_param(self, key):
        return super().get_param(self.section, key)

    def set_param(self, key=None, value=None):
        return super().set_param(self.section, key, value)

    def dict(self):
        return super().dict(self.section)
    """


    def initialize_parameters(self,
                              pathToModelDir,
                              default_config='default.cfg',
                              default_model=None,
                              additional_cli_section=None,
                              additional_definitions=None,
                              required=None):
        """Initialize Command Line Interface and Config."""
        self.logger.debug(
            f"Initializing parameters for {self.section}.")

        if additional_cli_section is None:
            additional_cli_section = 'Additional Parameters'

        print("additional_definitions(stage):", additional_definitions)
        if additional_definitions is not None:
            if isinstance(additional_definitions, str) or isinstance(additional_definitions, Path):
                additional_definitions = self.load_parameter_definitions(
                    additional_definitions)
            self.cli.set_command_line_options(
                additional_definitions, f'{additional_cli_section} options')

        p = super().initialize_parameters(pathToModelDir=pathToModelDir,
                                          section=self.section,
                                          default_config=default_config,
                                          default_model=default_model,
                                          additional_definitions=self.options,
                                          required=required)

        self.logger.setLevel(self.log_level)
        return p


class PreprocessConfig(SectionConfig):
    """Class to handle configuration files for Preprocessing."""

    def __init__(self) -> None:
        super().__init__('Preprocess', improve_preprocess_conf)


class TrainConfig(SectionConfig):
    """Class to handle configuration files for Training."""

    def __init__(self) -> None:
        super().__init__('Train', improve_train_conf)


class InferConfig(SectionConfig):
    """Class to handle configuration files for Inference."""

    def __init__(self) -> None:
        super().__init__('Infer', improve_infer_conf)


if __name__ == "__main__":
    p = PreprocessConfig()
    p.initialize_parameters(pathToModelDir=".")
    print(p.dict())

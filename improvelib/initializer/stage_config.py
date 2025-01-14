"""Module for configuration classes for different application stages.

This module defines configuration classes for preprocessing, training, and inference 
stages. It extends the base Config class to handle stage-specific configuration 
parameters and command-line interface options.

Classes:
    SectionConfig(Config): Base class for handling configuration for different sections.
    PreprocessConfig(SectionConfig): Configuration for the preprocessing stage.
    TrainConfig(SectionConfig): Configuration for the training stage.
    InferConfig(SectionConfig): Configuration for the inference stage.
"""

import logging
import os
from pathlib import Path
import pprint
from typing import Union

from improvelib.initializer.cli_params_def import (
    improve_basic_conf,
    improve_infer_conf,
    improve_preprocess_conf,
    improve_train_conf,
)
from improvelib.initializer.config import Config
from improvelib.utils import build_paths

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

printfn = pprint.PrettyPrinter(indent=4).pformat


class SectionConfig(Config):
    """Base class for handling section-specific configurations.

    This class extends the base Config class to provide additional functionality
    for managing configuration parameters specific to different stages of the
    application workflow.
    
    Attributes:
        section (str): The name of the section (e.g., "Preprocess", "Train").
        options (list): Combined list of global and stage-specific configuration parameters.

    Args:
        section (str): The section name.
        stage_config_parameters (list): Configuration parameters for the stage.
    """

    def __init__(self, section: str, stage_config_parameters: list) -> None:
        """Initializes the SectionConfig with section-specific parameters.
        
        Sets up logging for the section, combines basic and stage-specific configuration
        parameters, and configures command line options.

        Args:
            section (str): The section name.
            stage_config_parameters (list): Configuration parameters for the stage.
        """
        super().__init__()
        self.section = section
        self.logger = logging.getLogger(self.section)
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.INFO))

        # check usage of self.options
        self.options = improve_basic_conf + stage_config_parameters
        self.cli.set_command_line_options(
            improve_basic_conf, 'IMPROVE options')
        self.cli.set_command_line_options(
            options=stage_config_parameters, group=f'{self.section} stage options')

        # Add options for Benchmark Data Format
        p = self.cli.parser

    def initialize_parameters(self,
                              pathToModelDir: str,
                              default_config: str = 'default.cfg',
                              additional_cli_section: str = None,
                              additional_definitions: Union[str, Path] = None,
                              required: Union[list, None] = None) -> dict:
        """Initialize command line interface and configuration parameters.

        Args:
            pathToModelDir (str): Path to the model directory.
            default_config (str): Default configuration file. Defaults to 'default.cfg'.
            additional_cli_section (str): Additional CLI section name. Defaults to None.
            additional_definitions (str or Path): Additional parameter definitions. 
                Defaults to None.
            required (list): Required parameters. Defaults to None.

        Returns:
            dict: Dictionary of initialized parameters.
        """
        self.logger.debug(f"Initializing parameters for {self.section}.")

        if additional_cli_section is None:
            additional_cli_section = 'Additional Parameters'

        self.logger.debug("additional_definitions(stage): %s", printfn(additional_definitions))
        if additional_definitions is not None:
            if isinstance(additional_definitions, (str, Path)):
                additional_definitions = self.load_parameter_definitions(
                    additional_definitions)
            self.cli.set_command_line_options(
                additional_definitions, f'{additional_cli_section} options')
            self.options += additional_definitions

        p = super().initialize_parameters(pathToModelDir=pathToModelDir,
                                          section=self.section,
                                          default_config=default_config,
                                          additional_definitions=self.options,
                                          required=required)

        if self.section.lower() == 'preprocess':
            p = build_paths(p)

        self.logger.setLevel(self.log_level)
        return p


class PreprocessConfig(SectionConfig):
    """Configuration for the preprocessing stage.

    This class extends SectionConfig to handle configuration parameters
    specific to the preprocessing stage of the application workflow.
    """

    def __init__(self) -> None:
        """Initializes the PreprocessConfig with default preprocessing parameters."""
        super().__init__('Preprocess', improve_preprocess_conf)


class TrainConfig(SectionConfig):
    """Configuration for the training stage.

    This class extends SectionConfig to handle configuration parameters
    specific to the training stage of the application workflow.
    """

    def __init__(self) -> None:
        """Initializes the TrainConfig with default training parameters."""
        super().__init__('Train', improve_train_conf)


class InferConfig(SectionConfig):
    """Configuration for the inference stage.

    This class extends SectionConfig to handle configuration parameters
    specific to the inference stage of the application workflow.
    """

    def __init__(self) -> None:
        """Initializes the InferConfig with default inference parameters."""
        super().__init__('Infer', improve_infer_conf)


if __name__ == "__main__":
    # This block is for testing and debugging purposes.
    # It demonstrates how to initialize and use the PreprocessConfig class.
    p = PreprocessConfig()
    p.initialize_parameters(pathToModelDir=".")
    print(p.dict())

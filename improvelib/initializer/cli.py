"""Command-line interface operations for the IMPROVE model.

This module provides functionality to parse command-line arguments, set logging levels,
and manage configuration files for different stages of the ML pipeline.

Classes:
    CLI: Manages command-line argument parsing and handling.
"""

import argparse
import logging
import os
import pprint

from improvelib.utils import parse_from_dictlist

printfn = pprint.PrettyPrinter(indent=4).pformat


class CLI:
    """Handles command-line argument parsing and configuration.

    This class provides methods to parse and manage command-line arguments
    for the IMPROVE model, including setting default options and retrieving
    configuration files.

    Attributes:
        parser (ArgumentParser): The argument parser instance.
        logger (Logger): Logger instance for CLI operations.
        args: Parsed command line arguments.
        parser_params (dict): Dictionary of parsed arguments.
        default_params (dict): Dictionary of default parameter values.
        cli_explicit: Explicitly set CLI parameters.
        cli_params (dict): Dictionary of CLI parameters.
    """

    def __init__(self) -> None:
        """Initialize the CLI with default settings."""
        # Default format for logging
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        # Initialize parser
        self.parser = argparse.ArgumentParser(
            description='IMPROVE Command Line Parser',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Initialize logger
        self.logger = logging.getLogger('CLI')

        # Command line options after parsing
        self.args = None  # placeholder for args from argparse
        self.parser_params = None  # dict of args
        self.default_params = None  # dict of defaults for the parameters
        self.cli_explicit = None
        self.cli_params = {}

        # Set logger level
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.DEBUG))

        # Set common options for all model scripts
        common_options = self.parser.add_argument_group('Standard Options')
        common_options.add_argument(
            '-i', '--input_dir',
            metavar='DIR',
            type=str,
            dest="input_dir",
            default=os.getenv("IMPROVE_INPUT_DIR", "./"),
            help=(
                'Base directory for input data. Defaults to IMPROVE_DATA_DIR or '
                'the current working directory if not specified. All additional '
                'input paths are relative to this directory.'
            )
        )
        common_options.add_argument(
            '-o', '--output_dir',
            metavar='DIR',
            type=str,
            dest="output_dir",
            default=os.getenv("IMPROVE_OUTPUT_DIR", "./"),
            help=(
                'Base directory for output data. Defaults to IMPROVE_OUTPUT_DIR '
                'or the current working directory if not specified. All additional '
                'relative output paths will be placed into this directory.'
            )
        )
        common_options.add_argument(
            '--log_level',
            metavar='LEVEL',
            type=str,
            dest="log_level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
            default=os.getenv("IMPROVE_LOG_LEVEL", "WARNING"),
            help=(
                "Set the logging level. Defaults to WARNING. Available levels: "
                "DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET."
            )
        )
        common_options.add_argument(
            '--config_file',
            metavar='INI_FILE',
            dest="config_file",
            type=str,
            default=None,
            help=(
                "Path to a config file in INI format. Supports all command-line "
                "options. Command-line values override those in the config file."
            )
        )

    def set_command_line_options(self, options: list = [], group: str = None) -> None:
        """Set command line options, safeguarding standard options.

        Args:
            options (list): List of dictionaries defining command line options.
            group (str): Name of the argument group to add options to. Defaults to None.
        """
        self.logger.debug("Setting Command Line Options")
        self.logger.debug(f"Group: {group}")
        if not options:
            self.logger.warning("No options provided. Ignoring.")
            return

        # Find and remove duplicates
        unique_options = {}
        for d in options:
            if d['name'] not in unique_options:
                unique_options[d['name']] = d
            else:
                self.logger.warning(
                    "Found duplicate option %s in options. Removing duplicate", d['name']
                )

        # Create list of unique options
        options = list(unique_options.values())

        predefined_options = [o.lstrip('-')
                              for o in self.parser._option_string_actions]

        new_options = []

        for d in options:
            if d['name'] in predefined_options:
                self.logger.warning(
                    "Found %s in options. This option is predefined and cannot "
                    "be overwritten.", d['name']
                )
                self.logger.debug("Removing %s from options", d['name'])
                options.remove(d)
            else:
                self.logger.debug("Adding %s to new options", d['name'])
                new_options.append(d)

        self.logger.debug("Unique Options:\n%s", printfn(new_options))

        if group:
            group = self.parser.add_argument_group(group)
            self.logger.debug(f"Setting Group to {group}")
            parse_from_dictlist(new_options, group)
        else:
            parse_from_dictlist(new_options, self.parser)

    def get_command_line_options(self) -> dict:
        """Get command line options.

        Returns:
            dict: Dictionary of parsed command line arguments.
        """
        self.logger.debug("Getting Command Line Options")
        self.args = self.parser.parse_args()
        self.params = vars(self.args)
        self.cli_params = self.params

        return self.params

    def _check_option(self, option: str) -> bool:
        """Check if an option is valid.

        Args:
            option (str): Option to check.

        Returns:
            bool: True if option is valid, False otherwise.
        """
        pass

    def get_config_file(self) -> str:
        """Get the config file path from command line arguments.

        Expects --config_file option to be present in command line arguments.

        Returns:
            str: Path to the config file.
        """
        self.logger.debug("Getting the config file from the command line.")

        # Create a new parser to get the config file
        cfg_parser = argparse.ArgumentParser(
            description='Get the config file from command line.',
            add_help=False,
        )
        cfg_parser.add_argument(
            '--config_file',
            metavar='INI_FILE',
            type=str,
            dest="config_file",
            default=None
        )

        # Parse the command line arguments
        args_tmp = cfg_parser.parse_known_args()

        # Get the config file
        config_file = args_tmp[0].config_file

        self.logger.debug("Config file: %s", config_file)
        return config_file


if __name__ == "__main__":
    # This block is for testing/debugging purposes
    cli = CLI()
    defaults = [{'action': 'store', 'choices': [
        'A', 'B', 'C'], 'type': str, 'name': "dest"}]
    cli.set_command_line_options(options=defaults)
    cli.get_command_line_options()
    cfg = cli.config("Preprocess")

    for k in cli.params:
        print("\t".join([k, cli.params[k]]))
    print(cfg.dict(section="Preprocess"))
    setattr(cfg, "version", "0.1")
    print(cfg.version)

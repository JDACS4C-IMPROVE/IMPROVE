import argparse
import logging
import os
import pprint
import sys

# from improve import config as BaseConfig
from improvelib.utils import parse_from_dictlist

printfn = pprint.PrettyPrinter(indent=4).pformat


class CLI:
    """Base Class for Command Line Options"""

    def __init__(self):

        # Default format for logging
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        # Class attributes and defautl values
        # Initialize parser
        self.parser = argparse.ArgumentParser(
            description='IMPROVE Command Line Parser',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Initialize logger
        self.logger = logging.getLogger('CLI')

        # Command line options after parsing, results of self.parser.parse_args()
        self.args = None  # placeholder for args from argparse
        self.parser_params = None  # dict of args
        self.default_params = None # dict of defaults for the parameters
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
            default=os.getenv("IMPROVE_INPUT_DIR" , "./"),
            help='Base directory for input data. Default is IMPROVE_DATA_DIR \
                 or if not specified current working directory. All additional \
                 input pathes will be relative to the base input directory.'
        )
        common_options.add_argument(
            '-o', '--output_dir',
            metavar='DIR',
            type=str,
            dest="output_dir",
            default=os.getenv("IMPROVE_OUTPUT_DIR" , "./"), 
            help='Base directory for output data. Default is IMPROVE_OUTPUT_DIR \
                 or if not specified current working directory. All additional \
                 relative output pathes will be placed into the base output \
                 directory.'
        )
        common_options.add_argument(
            '--log_level',
            metavar='LEVEL',
            type=str,
            dest="log_level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
            default=os.getenv("IMPROVE_LOG_LEVEL", "WARNING"),
            help="Set log levels. Default is WARNING. Levels are: DEBUG, INFO, \
                 WARNING, ERROR, CRITICAL, NOTSET"
        ) 
        common_options.add_argument(
            '--config_file',
            metavar='INI_FILE',
            dest="config_file", 
            type=str,
            default=None,
            help="Config file in INI format. Supports all command line options. Values from the command line will overwrite values from the config file."
        )


    def set_command_line_options(self, options=[], group=None):
        """Set Command Line Options, saveguard standard options."""
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
                    "Found duplicate option %s in options. Removing duplicate", d['name'])

        # Create list of unique options
        options = list(unique_options.values())

        predefined_options = [o.lstrip('-')
                              for o in self.parser._option_string_actions]

        new_options = []
        # ['input_dir', 'output_dir', 'log_level', 'config_file']

        for d in options:
            if d['name'] in predefined_options:
                self.logger.warning(
                    "Found %s in options. This option is predefined and can \
                    not be overwritten.", d['name'])
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


    def get_command_line_options(self):
        """Get Command Line Options"""

        self.logger.debug("Getting Command Line Options")
        self.args = self.parser.parse_args()
        self.params = vars(self.args)
        self.cli_params = self.params

        return self.params


    def _check_option(self, option) -> bool:
        pass


    def get_config_file(self):
        """Get the config file from the command line. Expects --config_file option."""
        self.logger.debug("Getting the config file from the command line.")
        
        # Create a new parser to get the config file
        cfg_parser = argparse.ArgumentParser(
                        description='Get the config file from command line.',
                        add_help=False,)
        cfg_parser.add_argument(
            '--config_file', 
            metavar='INI_FILE', 
            type=str , 
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
    cli = CLI()
    defaults = [{'action': 'store', 'choices': [
        'A', 'B', 'C'], 'type': str, 'name': "dest"}]
    cli.set_command_line_options(options=defaults)
    cli.get_command_line_options()
    cfg=cli.config("Preprocess")

    for k in cli.params :
        print("\t".join([k , cli.params[k]]))
    print(cfg.dict(section="Preprocess"))
    setattr(cfg, "version" , "0.1")
    print(cfg.version)

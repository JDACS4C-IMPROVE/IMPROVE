import logging
import argparse
from argparse import Namespace
import copy
import os
import sys
import numpy as np # for candle functions, move later

# from candle import parse_from_dictlist
# from improve import config as BaseConfig

#TODO: move this to utils
class ListOfListsAction(argparse.Action):
    """This class extends the argparse.Action class by instantiating an
    argparser that constructs a list-of-lists from an input (command-line
    option or argument) given as a string."""

    def __init__(self, option_strings: str, dest, type, **kwargs):
        """Initialize a ListOfListsAction object. If no type is specified, an
        integer is assumed by default as the type for the elements of the list-
        of-lists.

        Parameters
        ----------
        option_strings : string
            String to parse
        dest : object
            Object to store the output (in this case the parsed list-of-lists).
        type : data type
            Data type to decode the elements of the lists.
            Defaults to np.int32.
        kwargs : object
            Python object containing other argparse.Action parameters.
        """

        super(ListOfListsAction, self).__init__(option_strings, dest, **kwargs)
        self.dtype = type
        if self.dtype is None:
            self.dtype = np.int32

    def __call__(self, parser, namespace, values, option_string=None):
        """This function overrides the __call__ method of the base
        argparse.Action class.

        This function implements the action of the ListOfListAction
        class by parsing an input string (command-line option or argument)
        and maping it into a list-of-lists. The resulting list-of-lists is
        added to the namespace of parsed arguments. The parsing assumes that
        the separator between lists is a colon ':' and the separator inside
        the list is a comma ','. The values of the list are casted to the
        type specified at the object initialization.

        Parameters
        ----------
        parser : ArgumentParser object
            Object that contains this action
        namespace : Namespace object
            Namespace object that will be returned by the parse_args()
            function.
        values : string
            The associated command-line arguments converted to string type
            (i.e. input).
        option_string : string
            The option string that was used to invoke this action. (optional)
        """

        decoded_list = []
        removed1 = values.replace("[", "")
        removed2 = removed1.replace("]", "")
        out_list = removed2.split(":")

        for line in out_list:
            in_list = []
            elem = line.split(",")
            for el in elem:
                in_list.append(self.dtype(el))
            decoded_list.append(in_list)

        setattr(namespace, self.dest, decoded_list)

def parse_from_dictlist(dictlist, parser):
    """
    Functionality to parse options.

    :param List pardict: Specification of parameters
    :param ArgumentParser parser: Current parser

    :return: consolidated parameters
    :rtype: ArgumentParser
    """

    for d in dictlist:
        if "type" not in d:
            d["type"] = None
        # print(d['name'], 'type is ', d['type'])

        if "default" not in d:
            d["default"] = argparse.SUPPRESS

        if "help" not in d:
            d["help"] = ""

        if "abv" not in d:
            d["abv"] = None

        if "action" in d:  # Actions
            if (
                d["action"] == "list-of-lists"
            ):  # Non standard. Specific functionallity has been added
                d["action"] = ListOfListsAction
                if d["abv"] is None:
                    parser.add_argument(
                        "--" + d["name"],
                        dest=d["name"],
                        action=d["action"],
                        type=d["type"],
                        default=d["default"],
                        help=d["help"],
                    )
                else:
                    parser.add_argument(
                        "-" + d["abv"],
                        "--" + d["name"],
                        dest=d["name"],
                        action=d["action"],
                        type=d["type"],
                        default=d["default"],
                        help=d["help"],
                    )
            elif (d["action"] == "store_true") or (d["action"] == "store_false"):
                raise Exception(
                    "The usage of store_true or store_false cannot be undone in the command line. Use type=str2bool instead."
                )
            else:
                if d["abv"] is None:
                    parser.add_argument(
                        "--" + d["name"],
                        action=d["action"],
                        default=d["default"],
                        help=d["help"],
                        type=d["type"],
                    )
                else:
                    parser.add_argument(
                        "-" + d["abv"],
                        "--" + d["name"],
                        action=d["action"],
                        default=d["default"],
                        help=d["help"],
                        type=d["type"],
                    )
        else:  # Non actions
            if "nargs" in d:  # variable parameters
                if "choices" in d:  # choices with variable parameters
                    if d["abv"] is None:
                        parser.add_argument(
                            "--" + d["name"],
                            nargs=d["nargs"],
                            choices=d["choices"],
                            default=d["default"],
                            help=d["help"],
                        )
                    else:
                        parser.add_argument(
                            "-" + d["abv"],
                            "--" + d["name"],
                            nargs=d["nargs"],
                            choices=d["choices"],
                            default=d["default"],
                            help=d["help"],
                        )
                else:  # Variable parameters (free, no limited choices)
                    if d["abv"] is None:
                        parser.add_argument(
                            "--" + d["name"],
                            nargs=d["nargs"],
                            type=d["type"],
                            default=d["default"],
                            help=d["help"],
                        )
                    else:
                        parser.add_argument(
                            "-" + d["abv"],
                            "--" + d["name"],
                            nargs=d["nargs"],
                            type=d["type"],
                            default=d["default"],
                            help=d["help"],
                        )
            elif "choices" in d:  # Select from choice (fixed number of parameters)
                if d["abv"] is None:
                    parser.add_argument(
                        "--" + d["name"],
                        choices=d["choices"],
                        default=d["default"],
                        help=d["help"],
                    )
                else:
                    parser.add_argument(
                        "-" + d["abv"],
                        "--" + d["name"],
                        choices=d["choices"],
                        default=d["default"],
                        help=d["help"],
                    )
            else:  # Non an action, one parameter, no choices
                # print('Adding ', d['name'], ' to parser')
                if d["abv"] is None:
                    parser.add_argument(
                        "--" + d["name"],
                        type=d["type"],
                        default=d["default"],
                        help=d["help"],
                    )
                else:
                    parser.add_argument(
                        "-" + d["abv"],
                        "--" + d["name"],
                        type=d["type"],
                        default=d["default"],
                        help=d["help"],
                    )

    return parser



class CLI:
    """Base Class for Command Line Options"""

    def __init__(self):

        # Default format for logging
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        # Class attributes and defautl values
        # Initialize parser
        self.parser = argparse.ArgumentParser(
            description='IMPROVE Command Line Parser')

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

    """        # Set common options for all model scripts
        common_options = self.parser.add_argument_group('Standard Options')
        common_options.add_argument('-i', '--input_dir', metavar='DIR', type=str, dest="input_dir",
                                  default=os.getenv("IMPROVE_INPUT_DIR" , "./"), 
                                  help='Base directory for input data. Default is IMPROVE_DATA_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        common_options.add_argument('-o', '--output_dir', metavar='DIR', type=str, dest="output_dir",
                                  default=os.getenv("IMPROVE_OUTPUT_DIR" , "./"), 
                                  help='Base directory for output data. Default is IMPROVE_OUTPUT_DIR or if not specified current working directory. All additional relative output pathes will be placed into the base output directory.')
        common_options.add_argument('--log_level', metavar='LEVEL', type=str, dest="log_level", 
                                  choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
                                  default=os.getenv("IMPROVE_LOG_LEVEL", "WARNING"), help="Set log levels. Default is WARNING. Levels are:\
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET") 
        common_options.add_argument('-cfg', '--config_file', metavar='INI_FILE', dest="config_file", 
                                  type=str,
                                  default=None, help="Config file in INI format.")  """

    def set_command_line_options(self, options=[], group=None):
        """Set Command Line Options, saveguard standard options."""
        self.logger.debug("Setting Command Line Options")
        self.logger.debug(f"Group: {group}")
        if not options:
            self.logger.warning("No options provided. Ignoring.")
            return

        predefined_options = [o.lstrip('-')
                              for o in self.parser._option_string_actions]
        # ['input_dir', 'output_dir', 'log_level', 'config_file']
        for o in predefined_options:
            # check if o is the value of name in one of the dicts in options
            for d in options:
                if o == d['name']:
                    self.logger.warning(
                        "Found %s in options. This option is predifined and can not be overwritten.", o)
                    self.logger.debug("Removing %s from options", o)
                    options.remove(d)

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

        # From Candle, can't handle bool, need to fork if we want to support argument groups
        if group:
            group = self.parser.add_argument_group(group)
            self.logger.debug(f"Setting Group to {group}")
            parse_from_dictlist(options, group)
        else:
            parse_from_dictlist(options, self.parser)

    def get_explicit_cli(self):
        self.logger.debug("Determining Options set by the Command Line")
        parser_copy = copy.deepcopy(self.parser)
        parser_copy.set_defaults(**{x:None for x in vars(self.args)})
        args_copy = parser_copy.parse_args()
        explicit_cli = Namespace(**{key:(value is not None) for key, value in vars(args_copy).items()})
        print("explicit cli:", vars(explicit_cli))
        # True if it was set by command line
        return vars(explicit_cli)

    def get_command_line_options(self):
        """Get Command Line Options"""

        self.logger.debug("Getting Command Line Options")
        print("here2")
        self.args = self.parser.parse_args()
        self.parser_params = vars(self.args)
        self.default_params = vars(self.parser.parse_args([]))
        self.cli_explicit = self.get_explicit_cli()
        print("here")
        print("self.cli_explicit:", self.cli_explicit)
        for explicit_key in self.cli_explicit:
            if self.cli_explicit[explicit_key]:
                self.cli_params[explicit_key] = self.parser_params[explicit_key]
        print("self.cli_params:", self.cli_params)


    def _check_option(self, option) -> bool:
        pass

    # def config(self, section) -> BaseConfig :
    #     cfg=BaseConfig.Config()
    #     if self.params['config_file']:
    #         if os.path.isfile(self.params['config_file']) :
    #             self.logger.info('Loading Config from %s', self.params['config_file'])
    #             cfg.file = self.params['config_file']
    #             cfg.load_config()
    #         else:
    #             self.logger.critical("Can't load Config from %s", self.params['config_file'])
    #     else:
    #         self.logger.debug('No config file')

    #     # Set params in config
    #     for k in self.params :
    #         (value,error) = cfg.param(section=section, key=k , value=self.params[k])

    #     return cfg

    def initialize_parameters(self,
                              pathToModelDir,
                              default_model=None,
                              additional_definitions=None,
                              required=None,):
        pass


if __name__ == "__main__":
    cli = CLI()
    defaults = [{'action': 'store', 'choices': [
        'A', 'B', 'C'], 'type': str, 'name': "dest"}]
    cli.set_command_line_options(options=defaults)
    cli.get_command_line_options()
    # cfg=cli.config("Preprocess")

    # for k in cli.params :
    #     print("\t".join([k , cli.params[k]]))
    # print(cfg.dict(section="Preprocess"))
    # setattr(cfg, "version" , "0.1")
    # print(cfg.version)

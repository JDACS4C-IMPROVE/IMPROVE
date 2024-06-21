from __future__ import absolute_import

import argparse
import os
import sys
import warnings
from pprint import PrettyPrinter, pprint

import numpy as np

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from typing import Any, List, Optional, Set, Type, Union

from .file_utils import directory_tree_from_parameters
from .helper_utils import str2bool

# Seed for random generation -- default value
DEFAULT_SEED = 7102
DEFAULT_TIMEOUT = -1  # no timeout
DEFAULT_DATATYPE = np.float32


class ArgumentStruct:
    """
    Class that converts a python dictionary into an object with named
    entries given by the dictionary keys.

    This structure simplifies the calling convention for accessing the
    dictionary values (corresponding to problem parameters). After the
    object instantiation both modes of access (dictionary or object
    entries) can be used.
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)


class ListOfListsAction(argparse.Action):
    """This class extends the argparse.Action class by instantiating an
    argparser that constructs a list-of-lists from an input (command-line
    option or argument) given as a string."""

    def __init__(self, option_strings: str, dest, type: Any, **kwargs):
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


class ParseDict(TypedDict):
    """Definition of the dictionary structure expected for the parsing of
    parameters."""

    name: str
    # ABV = abbreviated form
    abv: Optional[str]
    action: Union[str, Type[ListOfListsAction]]
    type: Optional[Any]
    default: Any
    help: str
    choices: List[str]
    nargs: str


class ConfigDict(TypedDict):
    """Definition of the dictionary structure expected for the configuration of
    parameters."""

    config_file: str
    data_type: str
    rng_seed: float
    train_bool: bool
    eval_bool: bool
    timeout: int
    gpus: Union[List[int], int]
    profiling: bool
    save_path: str
    model_name: str
    home_dir: str
    train_data: str
    val_data: str
    test_data: str
    output_dir: str
    data_url: str
    experiment_id: str
    run_id: str
    verbose: bool
    logfile: str
    scaling: str
    shuffle: bool
    feature_subsample: int
    dense: Union[List[int], int]
    conv: Union[List[int], int]
    locally_connected: bool
    activation: str
    out_activation: str
    lstm_size: int
    recurrent_dropout: float
    dropout: float
    pool: int
    batch_normalization: bool
    loss: str
    optimizer: str
    metrics: str
    epochs: int
    batch_size: int
    learning_rate: float
    early_stop: bool
    momentum: float
    initialization: str
    val_split: float
    train_steps: int
    val_steps: int
    test_steps: int
    train_samples: int
    val_samples: int
    clr_flag: bool
    clr_mode: str
    clr_base_lr: float
    clr_max_lr: float
    clr_gamma: float
    ckpt_restart_mode: str
    ckpt_checksum: bool
    ckpt_skip_epochs: int
    ckpt_directory: str
    ckpt_save_best: bool
    ckpt_save_best_metric: str
    ckpt_save_weights_only: bool
    ckpt_save_interval: int
    ckpt_keep_mode: str
    ckpt_keep_limit: int

improve_conf = [
    {
        "name": "model_name",
        "type": str,
        "default": "ABC",
        "help": "Name of curated model.",
    },
]

registered_conf = [improve_conf,]

def extract_keywords(lst_dict, kw):
    """Extract the value associated to a specific keyword in a list of
    dictionaries. Returns the list of values extracted from the keywords.

    Parameters
    ----------
    lst_dict : python list of dictionaries
       list to extract keywords from
    kw : string
       keyword to extract from dictionary
    """
    lst = [di[kw] for di in lst_dict]
    return lst


# Extract list of parameters in registered configuration
PARAMETERS_CANDLE = [
    item for lst in registered_conf for item in extract_keywords(lst, "name")
]

CONFLICT_LIST = [["clr_flag", "warmup_lr"], ["clr_flag", "reduce_lr"]]


def check_flag_conflicts(params: ConfigDict):
    """
    Check if parameters that must be exclusive are used in conjunction. The
    check is made against CONFLICT_LIST, a global list that describes parameter
    pairs that should be exclusive. Raises an exception if pairs of parameters
    in CONFLICT_LIST are specified simulataneously.

    :param Dict params: list to extract keywords from
    """
    key_set: Set[str] = set(params.keys())
    # check for conflicts
    # conflict_flag = False
    # loop over each set of mutually exclusive flags
    # if any set conflicts exit program
    for flag_list in CONFLICT_LIST:
        flag_count = 0
        for i in flag_list:
            if i in key_set:
                if params[i] is True:
                    flag_count += 1
        if flag_count > 1:
            raise Exception(
                "ERROR ! Conflict in flag specification. These flags should not be used together: "
                + str(sorted(flag_list))
                + "... Exiting"
            )


def check_file_parameters_exists(
    params_parser: ConfigDict, params_benchmark: ConfigDict, params_file: ConfigDict
):
    """Functionality to verify that the parameters defined in the configuration
    file are recognizable by the command line parser (i.e. no uknown keywords
    are used in the configuration file).

    Parameters
    ----------
    params_parser : python dictionary
        Includes parameters set via the command line.
    params_benchmark : python list
        Includes additional parameters defined in the benchmark.
    params_file : python dictionary
        Includes parameters read from the configuration file.

        Global:
        PARAMETERS_CANDLE : python list
            Includes all the core keywords that are specified in CANDLE.
    """
    # Get keywords from arguments coming via command line (and CANDLE supervisor)
    args_dict = vars(params_parser)
    args_set = set(args_dict.keys())
    # Get keywords from benchmark definition
    bmk_keys: List[str] = []
    for item in params_benchmark:
        bmk_keys.append(item["name"])
    bmk_set = set(bmk_keys)
    # Get core CANDLE keywords
    candle_set = set(PARAMETERS_CANDLE)
    # Consolidate keywords from CANDLE core, command line, CANDLE supervisor and benchmark
    candle_set = candle_set.union(args_set)
    candle_set = candle_set.union(bmk_set)
    # Get keywords used in config_file
    file_set = set(params_file.keys())
    # Compute keywords that come from the config_file that are not in the CANDLE specs
    diff_set = file_set.difference(candle_set)

    if len(diff_set) > 0:
        message = (
            "These keywords used in the configuration file are not defined in CANDLE: "
            + str(sorted(diff_set))
        )
        warnings.warn(message, RuntimeWarning)


def finalize_parameters(bmk):
    """
    Utility to parse parameters in common as well as parameters particular
    to each benchmark.

    :param Benchmark bmk: Object that has benchmark filepaths and specifications

    :return: Dictionary with all the parameters necessary to run the benchmark.\
        Command line overwrites config file specifications
    """

    # Parse common and benchmark parameters
    bmk.parse_parameters()

    # print('Args:', args)
    # Get parameters from configuration file
    # Reads parameter subset, just checking if a config_file has been set
    # by comand line (the parse_known_args() function allows a partial
    # parsing)
    aux = bmk.parser.parse_known_args()
    try:  # Try to get the 'config_file' option
        conffile_txt = aux[0].config_file
    except AttributeError:  # The 'config_file' option was not set by command-line
        conffile = bmk.conffile  # use default file
    else:  # a 'config_file' has been set --> use this file
        if os.path.isabs(conffile_txt):
            conffile = conffile_txt
        elif conffile_txt.startswith(
            "./"
        ):  # user is trying to use a builtin alternate model file
            conffile = os.path.join(bmk.file_path, conffile_txt)
        else:
            if os.environ["CANDLE_DATA_DIR"] is not None:
                conffile = os.path.join(os.environ["CANDLE_DATA_DIR"], conffile_txt)
            else:
                conffile = os.path.join(bmk.file_path, conffile_txt)
    # check conffile exists
    if not os.path.isfile(conffile):
        raise Exception(
            "ERROR ! Specified configuration file "
            + conffile
            + " not found ... Exiting"
        )
    else:
        print("Configuration file: ", conffile)

    fileParameters = bmk.read_config_file(
        conffile
    )  # aux.config_file)#args.config_file)
    # Get command-line parameters
    args = bmk.parser.parse_args()
    # print ('Params:', fileParameters)
    # Check keywords from file against CANDLE common and module definitions
    bmk_dict = bmk.additional_definitions
    check_file_parameters_exists(args, bmk_dict, fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = args_overwrite_config(args, fileParameters)
    # Check that required set of parameters has been defined
    bmk.check_required_exists(gParameters)
    print("Params:")
    pprint(gParameters)
    # Dump the parameters to the run directory
    final_params = os.path.join(gParameters["output_dir"], "final_params.txt")
    with open(final_params, "w") as fp:
        pp = PrettyPrinter(stream=fp)
        pp.pprint(gParameters)

    # Check that no keywords conflict
    check_flag_conflicts(gParameters)

    return gParameters


def args_overwrite_config(args, config: ConfigDict):
    """Overwrite configuration parameters with parameters specified via
    command-line.

    Parameters
    ----------
    args : ArgumentParser object
        Parameters specified via command-line
    config : python dictionary
        Parameters read from configuration file
    """

    params = config

    args_dict = vars(args)

    for key in args_dict.keys():
        # try casting here
        params[key] = args_dict[key]

    if "data_type" not in params:
        params["data_type"] = DEFAULT_DATATYPE
    else:
        if params["data_type"] in set(["f16", "f32", "f64"]):
            params["data_type"] = get_choice(params["data_type"])

    if "output_dir" not in params:
        params["data_dir"], params["output_dir"] = directory_tree_from_parameters(
            params
        )
    else:
        params["data_dir"], params["output_dir"] = directory_tree_from_parameters(
            params, params["output_dir"]
        )

    if "rng_seed" not in params:
        params["rng_seed"] = DEFAULT_SEED

    if "timeout" not in params:
        params["timeout"] = DEFAULT_TIMEOUT

    return params


def get_choice(name: str):
    """Maps name string to the right type of argument."""
    mapping = {}

    # dtype
    mapping["f16"] = np.float16
    mapping["f32"] = np.float32
    mapping["f64"] = np.float64

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mapping found for "{}"'.format(name))

    return mapped


def parse_from_dictlist(dictlist: List[ParseDict], parser):
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


def parse_common(parser):
    """Functionality to parse options.
    Parameters
    ----------
    parser : ArgumentParser object
        Current parser
    """

    for lst in registered_conf:
        parser = parse_from_dictlist(lst, parser)

    return parser

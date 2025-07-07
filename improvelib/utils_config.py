""" Basic definitions for IMPROVE framework. """

import argparse
import json
import os
import time
from pathlib import Path
# use NewType becuase TypeAlias is available from python 3.10
from typing import List, Set, Union, NewType, Dict, Optional, Any

import numpy as np
import pandas as pd





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
            # Select from choice (fixed number of parameters)
            elif "choices" in d:
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


def check_path(path: Path):
    if path.exists() == False:
        raise Exception(f"ERROR ! {path} not found.\n")


def build_paths(params: Dict):
    """ Build paths for raw_data, x_data, y_data, splits.
    These paths determine directories for a benchmark dataset.
    TODO: consider renaming to build_benchmark_data_paths()

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: updated dict of CANDLE/IMPROVE parameters and parsed values.
    """
    mainpath = Path(params["input_dir"])
    check_path(mainpath)

    # Raw data
    raw_data_path = mainpath
    params["raw_data_path"] = raw_data_path
    check_path(raw_data_path)

    x_data_path = raw_data_path / params["x_data_dir"]
    params["x_data_path"] = x_data_path
    check_path(x_data_path)

    y_data_path = raw_data_path / params["y_data_dir"]
    params["y_data_path"] = y_data_path
    check_path(y_data_path)

    splits_path = raw_data_path / params["splits_dir"]
    params["splits_path"] = splits_path
    check_path(splits_path)

    return params

class StoreIfPresent(argparse.Action):
    """
    This class allows to define an argument in argparse that keeps the default
    value empty and, if not passed by the user, the argument is not available
    in the parsed arguments. By default, argparse includes all defined arguments
    in the Namespace object returned by parse_args(), even if they are not
    provided by the user, assigning them the default value.

    This is primarily used with args that we plan to deprecate.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Args:
            parser (ArgumentParser object): Object that contains this action
            namespace (Namespace object): Namespace object that will be
                returned by the parse_args() function.
            values (str): The associated command-line arguments converted to
                string type (i.e. input).
            option_string (str): The option string that was used to invoke
                this action. (optional)
        """
        setattr(namespace, self.dest, values)

def cast_value(s):
    """Cast to numeric if possible
    Note: this doesn't seem to be used anywhere anymore.
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s  # Return the original string if it's neither int nor float
        

def create_outdir(outdir: Union[Path, str]):
    """ Create directory.

    Args:
        outdir (Path or str): dir path to create

    Returns:
        pathlib.Path: returns the created dir path

    Note: no longer used anywhere.
    """
    outdir = Path(outdir)
    if outdir.exists():
        print(f"Dir already exists: {outdir}")
    else:
        print(f"Creating dir: {outdir}")
        os.makedirs(outdir, exist_ok=True)
    check_path(outdir)
    return outdir

def save_subprocess_stdout(
    result,
    log_dir: Union[str, Path]='.',
    log_filename: Optional[str]='logs.txt'):
    """ Save the captured output from subprocess python package.
    Args:
        result: captured output from subprocess python package.
            E.g. result = subprocess.run(...)
        log_dir (str or Path): dir to save the logs
        log_filename (str): file name to save the logs

    Note: no longer used anywhere
    """
    result_file_name_stdout = log_dir / log_filename
    with open(result_file_name_stdout, 'w') as file:
        file.write(result.stdout)
    return True
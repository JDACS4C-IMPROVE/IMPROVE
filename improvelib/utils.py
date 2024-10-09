""" Basic definitions for IMPROVE framework. """

import argparse
import json
import os
import time
from pathlib import Path
# use NewType becuase TypeAlias is available from python 3.10
from typing import List, Set, Union, NewType, Dict, Optional

import numpy as np
import pandas as pd

from .metrics import compute_metrics


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
    """
    result_file_name_stdout = log_dir / log_filename
    with open(result_file_name_stdout, 'w') as file:
        file.write(result.stdout)
    return True


class Timer:
    """ Measure time. """
    def __init__(self):
        self.start = time.time()

    def timer_end(self):
        self.end = time.time()
        self.time_diff = self.end - self.start
        self.hours = int(self.time_diff // 3600)
        self.minutes = int((self.time_diff % 3600) // 60)
        self.seconds = self.time_diff % 60
        self.time_diff_dict = {'hours': self.hours,
                               'minutes': self.minutes,
                               'seconds': self.seconds}

    def display_timer(self, print_fn=print):
        self.timer_end()
        tt = self.time_diff_dict
        print(f"Elapsed Time: {self.hours:02}:{self.minutes:02}:{self.seconds:05}")
        return self.time_diff_dict

    def save_timer(self,
                   dir_to_save: Union[str, Path]='.',
                   filename: str='runtime.json',
                   extra_dict: Optional[Dict]=None):
        """ Save runtime to file. """
        if isinstance(extra_dict, dict):
            self.time_diff_dict.update(extra_dict)
        with open(Path(dir_to_save) / filename, 'w') as json_file:
            json.dump(self.time_diff_dict, json_file, indent=4)
        return True


def str2bool(v: str) -> bool:
    """
    This is taken from:
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-
    argparse Because type=bool is not interpreted as a bool and
    action='store_true' cannot be undone.

    :param string v: String to interpret

    :return: Boolean value. It raises and exception if the provided string cannot \
        be interpreted as a boolean type.

        - Strings recognized as boolean True : \
            'yes', 'true', 't', 'y', '1' and uppercase versions (where applicable).
        - Strings recognized as boolean False : \
            'no', 'false', 'f', 'n', '0' and uppercase versions (where applicable).
    :rtype: boolean
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def cast_value(s):
    """Cast to numeric if possbile"""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s  # Return the original string if it's neither int nor float


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

    # # ML data dir
    # ml_data_path = mainpath / params["ml_data_outdir"]
    # params["ml_data_path"] = ml_data_path
    # os.makedirs(ml_data_path, exist_ok=True)
    # check_path(ml_data_path)
    # os.makedirs(params["ml_data_outdir"], exist_ok=True)
    # check_path(params["ml_data_outdir"])

    # Models dir
    # os.makedirs(params["model_outdir"], exist_ok=True)
    # check_path(params["model_outdir"])

    # Infer dir
    # os.makedirs(params["infer_outdir"], exist_ok=True)
    # check_path(params["infer_outdir"])

    return params


def create_outdir(outdir: Union[Path, str]):
    """ Create directory.

    Args:
        outdir (Path or str): dir path to create

    Returns:
        pathlib.Path: returns the created dir path
    """
    outdir = Path(outdir)
    if outdir.exists():
        print(f"Dir already exists: {outdir}")
    else:
        print(f"Creating dir: {outdir}")
        os.makedirs(outdir, exist_ok=True)
    check_path(outdir)
    return outdir


# def create_ml_data_outdir(params: Dict):
#     """ Create directory to store data files for ML/DL models.
#     Used in *preprocess*.py
#     """
#     ml_data_dir = Path(params["ml_data_outdir"])
#     if ml_data_dir.exists():
#         print(f"ml_data_outdir already exists: {ml_data_dir}")
#     else:
#         print(f"Creating ml_data_outdir: {ml_data_dir}")
#         os.makedirs(ml_data_dir, exist_ok=True)
#     check_path(ml_data_dir)
#     return ml_data_dir


def get_file_format(file_format: Union[str, None] = None):
    """ Clean file_format.
    Exmamples of (input, return) pairs:
    input, return: "", ""
    input, return: None, ""
    input, return: "pt", ".pt"
    input, return: ".pt", ".pt"
    """
    file_format = "" if file_format is None else file_format
    if file_format != "" and "." not in file_format:
        file_format = "." + file_format
    return file_format


# def build_ml_data_name(params: Dict, stage: str, file_format: str=""):
def build_ml_data_file_name(data_format: str, stage: str):
    """ Returns name of the ML/DL data file. E.g., train_data.pt
    Used in *preprocess*.py*, *train*.py, and *infer*.py
    """
    data_file_format = get_file_format(file_format=data_format)
    ml_data_file_name = stage + "_" + "data" + data_file_format
    return ml_data_file_name


def build_model_path(model_file_name: str, model_file_format: str, model_dir: Union[Path, str]):
    """ Build path to save the trained model.
    Used in *train*.py and *infer*.py

    Args:
        model_file_name str: Name of model file.
        model_file_format: Type of file for model (e.g. '.pt').
        model_dir (Path or str): dir path to save the model

    Returns:
        pathlib.Path: returns the build model dir path
    """
    if model_file_format == "None":
        model_path = Path(model_dir) / \
        (model_file_name)
    else:
        standard_model_file_format = get_file_format(
        file_format=model_file_format)
        model_path = Path(model_dir) / \
        (model_file_name + standard_model_file_format)

    return model_path


# def create_model_outpath(params: Dict):
# # def create_model_outpath(params: Dict, model_dir):
#     """ Create path to save the trained model
#     Used in *train*.py
#     """
#     model_dir = Path(params["model_outdir"])
#     os.makedirs(model_dir, exist_ok=True)
#     check_path(model_dir)
#     # model_file_format = get_file_format(file_format=params["model_file_format"])
#     # model_path = model_dir / (params["model_file_name"] + model_file_format)
#     model_path = build_model_path(params, model_dir)
#     return model_path


def save_stage_ydf(ydf: pd.DataFrame, stage: str, output_dir: str):
    """ Save a subset of y data samples (rows of the input dataframe).
    The "subset" refers to one of the three stages involved in developing ML
    models, including: "train", "val", or "test".

    Args:
        ydf (pd.DataFrame): dataframe with y data samples
        stage (str) : "train", "val", or "test"
        output_dir str: Directory to save to.
    """
    ydf_fname = f"{stage}_y_data.csv"
    ydf_fpath = Path(output_dir) / ydf_fname
    ydf.to_csv(ydf_fpath, index=False)

    return None


def store_predictions_df(y_pred: np.array,
                         y_col_name: str,
                         stage: str,
                         output_dir: str,
                         input_dir: Optional[str] = None,
                         y_true: Optional[np.array] = None,
                         round_decimals: int = 4) -> None:

    """ Save predictions with accompanying dataframe.

    This allows to trace original data evaluated (e.g. drug and cell
    paris) if corresponding dataframe is available, in which case the
    whole structure as well as the model predictions are stored. If the
    dataframe is not available, only ground truth and model predictions are
    stored.

    Args:
        y_pred (array): Model predictions
        y_col_name (str): Name of the column in the y_data predicted on
        stage (str): specify if evaluation is with respect to val or test set
        output_dir (str): Directory to write results
        y_true (array): Ground truth
        input_dir (str): Directory where df with ground truth with metadata is stored
        round_decimals (int): Number of decimals in output
    """
    cast_ydata = np.float32

    # Put predictions in a df
    pred_col_name = y_col_name + "_pred" # define colname for predicted values
    pred_df = pd.DataFrame({pred_col_name: y_pred}) # create df
    pred_df = pred_df.astype({pred_col_name: cast_ydata}) # cast
    pred_df = pred_df.round({pred_col_name: round_decimals}) # round decimal

    # Add ground truth values if available to the pred_df
    if y_true is not None:
        # Check that y_true and y_pred dims match
        assert len(y_true) == len(y_pred), f"length mismatch of y_true \
            ({len(y_true)}) and y_pred ({len(y_pred)})"

        true_col_name = y_col_name + "_true"
        pred_df.insert(0, true_col_name, y_true, allow_duplicates=True) # add col to df
        pred_df = pred_df.astype({true_col_name: cast_ydata}) # cast
        pred_df = pred_df.round({true_col_name: round_decimals}) # round decimal

    # ydf refers to a file that can contain metadata of ydata and possibly the
    # ground truth values (e.g., metadata df that contains cancer ids, drug
    # ids, and the true response values)
    ydf_fname = f"{stage}_y_data.csv" # name of ydf if it exists
    ydf_out_fname = ydf_fname.split(".")[0] + "_predicted.csv" # fname for output ydf
    ydf_out_fpath = Path(output_dir) / ydf_out_fname # path for output ydf

    # Attempt to concatenate raw predictions with y dataframe (e.g., metadata
    # df that contains cancer ids, drug ids, and the true response values)
    # Check if ydf exists
    if (input_dir is not None) and (Path(input_dir) / ydf_fname).exists():
        ydf_fpath = Path(input_dir) / ydf_fname
        rsp_df = pd.read_csv(ydf_fpath)
        rsp_df = rsp_df.astype({y_col_name: cast_ydata}) # cast
        rsp_df = rsp_df.round({y_col_name: round_decimals}) # round decimal

        # Check if ground truth is available ydf
        if y_true is not None:
            # Check that ydf and ground truth dims match
            assert len(y_true) == rsp_df.shape[0], f"length mismatch of y_true \
                ({len(y_true)}) and loaded ydf ({ydf_fpath} ==> {rsp_df.shape[0]})"

            if y_col_name in rsp_df.columns:
                v1 = rsp_df[y_col_name].values
                v2 = pred_df[true_col_name].values
                # Check that values of ground truth in ydf and y_true actually match
                assert np.array_equal(v1, v2), "Loaded y data array is not \
                    equal to the true array"

        df = pd.concat([rsp_df, pred_df], axis=1)

    else:
        df = pred_df.copy()

    df.to_csv(ydf_out_fpath, index=False)  # Save predictions df

    return None


def compute_performance_scores(y_true: np.array,
                               y_pred: np.array,
                               stage: str, 
                               metric_type: str, 
                               output_dir: str):
    """Evaluate predictions according to specified metrics.

    Metrics are evaluated. Scores are stored in specified path and returned.


    :params array y_true: Array with ground truth values.
    :params array y_pred: Array with model predictions.
    :params str stage: String specified if evaluation is with respect to
            validation or testing set.
    :params str metric_type: Either classification or regression.
    :params str output_dir: Directory to write results.

    :return: Python dictionary with metrics evaluated and corresponding scores.
    :rtype: dict
    """
    # Compute multiple performance scores
    scores = compute_metrics(y_true, y_pred, metric_type)

    # Add val_loss metric
    #key = f"{stage}_loss"
    #scores[key] = scores[params["loss"]]

    scores_fname = f"{stage}_scores.json"
    scorespath = Path(output_dir) / scores_fname

    with open(scorespath, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    # TODO. do we still need to print IMPROVE_RESULT?
    if stage == "val":
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
        print("Validation scores:\n\t{}".format(scores))
    elif stage == "test":
        print("Inference scores:\n\t{}".format(scores))
    else:
        print("Invalid stage: must be 'val' or 'test'.")
    return scores


def check_path_and_files(folder_name: str, file_list: List, inpath: Path) -> Path:
    """Checks if a folder and its files are available in path.

    Returns a path to the folder if it exists or raises an exception if it does
    not exist, or if not all the listed files are present.

    :param string folder_name: Name of folder to look for in path.
    :param list file_list: List of files to look for in folder
    :param inpath: Path to look into for folder and files

    :return: Path to folder requested
    :rtype: Path
    """
    # TODO. this func is not currently used
    outpath = inpath / folder_name
    # Check if folder is in path
    if outpath.exists():
        # Make sure that the specified files exist
        for fn in file_list:
            auxdir = outpath / fn
            if auxdir.exists() == False:
                raise Exception(f"ERROR ! {fn} file not available.\n")
    else:
        raise Exception(f"ERROR ! {folder_name} folder not available.\n")

    return outpath

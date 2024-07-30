""" Basic definitions for IMPROVE framework. """

import os
import argparse
import json
from pathlib import Path
# use NewType becuase TypeAlias is available from python 3.10
from typing import List, Set, Union, NewType, Dict, Optional

import numpy as np
import pandas as pd

from .metrics import compute_metrics


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
def build_ml_data_name(params: Dict, stage: str):
    """ Returns name of the ML/DL data file. E.g., train_data.pt
    TODO: consider renaming build_ml_data_file_name()
    Used in *preprocess*.py*, *train*.py, and *infer*.py
    """
    # data_file_format = get_file_format(file_format=file_format)
    data_file_format = get_file_format(file_format=params["data_format"])
    ml_data_file_name = stage + "_" + "data" + data_file_format
    return ml_data_file_name


def build_model_path(params: Dict, model_dir: Union[Path, str]):
    """ Build path to save the trained model.
    Used in *train*.py and *infer*.py

    Args:
        params (dict): dict of IMPROVE/CANDLE parameters and parsed values.
        model_dir (Path or str): dir path to save the model

    Returns:
        pathlib.Path: returns the build model dir path
    """
    model_file_format = get_file_format(
        file_format=params["model_file_format"])
    model_path = Path(model_dir) / \
        (params["model_file_name"] + model_file_format)

    # # check for model_outdir and output_dir in params and use the one that is available
    # # this ensures backward compatibility with previous versions of framework.py
    # model_file_format = get_file_format(file_format=params["model_file_format"])
    # if "model_outdir" is params:
    #     model_dir = Path(model_dir)
    # elif "output_dir" in params:
    #     model_dir = Path(params["output_dir"])
    # else:
    #     raise Exception(
    #         "ERROR ! Neither 'ml_data_outdir' not 'output_dir' found in params.\n")
    # model_path = model_dir / (params["model_file_name"] + model_file_format)

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


def save_stage_ydf(ydf: pd.DataFrame, params: Dict, stage: str):
    """ Save a subset of y data samples (rows of the input dataframe).
    The "subset" refers to one of the three stages involved in developing ML
    models, including: "train", "val", or "test".

    ydf : dataframe with y data samples
    params : parameter dict
    stage (str) : "train", "val", or "test"
    """
    ydf_fname = f"{stage}_{params['y_data_suffix']}.csv"

    # check for ml_data_outdir and output_dir in params and use the one that is available
    # this ensures backward compatibility with previous versions of framework.py
    if "ml_data_outdir" in params:
        ydf_fpath = Path(params["ml_data_outdir"]) / ydf_fname
    elif "output_dir" in params:
        ydf_fpath = Path(params["output_dir"]) / ydf_fname
    else:
        raise Exception(
            "ERROR ! Neither 'ml_data_outdir' not 'output_dir' found in params.\n")

    ydf.to_csv(ydf_fpath, index=False)
    return None


def store_predictions_df(params: Dict,
                         y_true: np.array,
                         y_pred: np.array,
                         stage: str,
                         outdir: Union[Path, str],
                         round_decimals: int = 4):
    """Store predictions with accompanying data frame.

    This allows to trace original data evaluated (e.g. drug and cell
    combinations) if corresponding data frame is available, in which case
    the whole structure as well as the model predictions are stored. If
    the data frame is not available, only ground truth (read from the
    PyTorch processed data) and model predictions are stored. The ground
    truth available (from data frame or PyTorch data) is returned for
    further evaluations.

    ap: construct output file name as follows:

            [stage]_[params['y_data_suffix']]_

    :params Dict params: Dictionary of CANDLE/IMPROVE parameters read.
    :params Dict indtd: Dictionary specifying paths of input data.
    :params Dict outdtd: Dictionary specifying paths for ouput data.
    :params array y_true: Ground truth.
    :params array y_pred: Model predictions.

    :return: Arrays with ground truth. This may have been read from an
             input data frame or from a processed PyTorch data file.
    :rtype: np.array
    """
    # Check dimensions
    assert len(y_true) == len(
        y_pred), f"length of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) don't match"
    # print(len(y_true))
    # print(len(y_pred))

    # Define column names
    pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    true_col_name = params["y_col_name"] + "_true"

    # -----------------------------
    # Attempt to concatenate raw predictions with y dataframe (e.g., df that
    # contains cancer ids, drug ids, and the true response values)
    # TODO. f"{stage}_{params['y_data_stage_fname_suffix']}.csv"
    ydf_fname = f"{stage}_{params['y_data_suffix']}.csv"
    # ydf_fpath = Path(params[f"{stage}_ml_data_dir"]) / ydf_fname

    # check for ml_data_outdir and output_dir in params and use the one that is available
    # this ensures backward compatibility with previous versions of framework.py
    if f"{stage}_ml_data_dir" in params:
        ydf_fpath = Path(params[f"{stage}_ml_data_dir"]) / ydf_fname
    elif "output_dir" in params:
        ydf_fpath = Path(params["output_dir"]) / ydf_fname
    else:
        raise Exception(
            f"ERROR ! Neither '{stage}_ml_data_dir' not 'output_dir' found in params.\n")

    # output df fname
    ydf_out_fname = ydf_fname.split(
        ".")[0] + "_" + params["y_data_preds_suffix"] + ".csv"
    # ydf_out_fpath = Path(params["ml_data_outdir"]) / ydf_out_fname
    ydf_out_fpath = Path(outdir) / ydf_out_fname

    # if indtd["df"] is not None:
    if ydf_fpath.exists():
        rsp_df = pd.read_csv(ydf_fpath)

        # Check dimensions
        assert len(
            y_true) == rsp_df.shape[0], f"length of y_true ({len(y_true)}) and the loaded file ({ydf_fpath} --> {rsp_df.shape[0]}) don't match"

        # pred_df = pd.DataFrame(y_pred, columns=[pred_col_name])  # Include only predicted values
        # This includes only predicted values
        pred_df = pd.DataFrame({true_col_name: y_true, pred_col_name: y_pred})
        v1 = np.round(rsp_df[params["y_col_name"]].values.astype(np.float32),
                      decimals=round_decimals)
        v2 = np.round(pred_df[true_col_name].values.astype(np.float32),
                      decimals=round_decimals)
        # breakpoint()
        assert np.array_equal(
            v1, v2), "Loaded y data vector is not equal to the true vector"
        mm = pd.concat([rsp_df, pred_df], axis=1)
        mm = mm.astype({params["y_col_name"]: np.float32,
                        true_col_name: np.float32,
                        pred_col_name: np.float32})
        df = mm.round({true_col_name: round_decimals,
                       pred_col_name: round_decimals})
        df.to_csv(ydf_out_fpath, index=False)  # Save predictions dataframe
        # y_true_return = rsp_df[params["y_col_name"]].values # Read from data frame

    else:
        # Save only ground truth and predictions since did not load the corresponding dataframe
        # This includes true and predicted values
        df = pd.DataFrame({true_col_name: y_true, pred_col_name: y_pred})
        mm = df
        mm = mm.astype({true_col_name: np.float32,
                        pred_col_name: np.float32})
        df = mm.round({true_col_name: round_decimals,
                       pred_col_name: round_decimals})
        df.to_csv(ydf_out_fpath, index=False)
        # y_true_return = y_true

    # return y_true_return
    return None


def compute_performance_scores(params: Dict,
                               y_true: np.array,
                               y_pred: np.array,
                               stage: str,
                               outdir: Union[Path, str],
                               metrics: List):
    """Evaluate predictions according to specified metrics.

    Metrics are evaluated. Scores are stored in specified path and returned.

    :params array y_true: Array with ground truth values.
    :params array y_pred: Array with model predictions.
    :params listr metrics: List of strings with metrics to evaluate.
    :params Dict outdtd: Dictionary with path to store scores.
    :params str stage: String specified if evaluation is with respect to
            validation or testing set.

    :return: Python dictionary with metrics evaluated and corresponding scores.
    :rtype: dict
    """
    # Compute multiple performance scores
    scores = compute_metrics(y_true, y_pred, metrics)

    # Add val_loss metric
    key = f"{stage}_loss"
    # scores[key] = scores["mse"]
    scores[key] = scores[params["loss"]]

    # fname = f"val_{params['json_scores_suffix']}.json"
    scores_fname = f"{stage}_{params['json_scores_suffix']}.json"
    # scorespath = Path(params["ml_data_outdir"]) / scores_fname
    scorespath = Path(outdir) / scores_fname

    with open(scorespath, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    # TODO. do we still need to print IMPROVE_RESULT?
    if stage == "val":
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
        print("Validation scores:\n\t{}".format(scores))
    elif stage == "test":
        print("Inference scores:\n\t{}".format(scores))
    return scores


compute_performace_scores = compute_performance_scores  # for backwards compatibility


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

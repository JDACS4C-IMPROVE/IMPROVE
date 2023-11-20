""" Basic Definitions for IMPROVE Framework. """

import os
import argparse
from pathlib import Path
from typing import List, Set, NewType, Dict, Optional # use NewType becuase TypeAlias is available from python 3.10

import pandas as pd
import torch

# Check that environment variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception("ERROR ! Required system variable not specified.  \
                    You must define IMPROVE_DATA_DIR ... Exiting.\n")
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

SUPPRESS = argparse.SUPPRESS

import candle
str2bool = candle.str2bool
finalize_parameters = candle.finalize_parameters


# DataPathDict: TypeAlias = dict[str, Path]
DataPathDict = NewType("DataPathDict", Dict[str, Path])


# Parameters that are relevant to all IMPROVE models
improve_basic_conf = [
    {"name": "y_col_name",
     "type": str,
     "default": "auc",
     "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
    },
    {"name": "pred_col_name_suffix",
     "type": str,
     "default": "_pred",
     "help": "Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
    },

]

# Parameters that are relevant to all IMPROVE preprocessing scripts
improve_preprocess_conf = [
    {"name": "raw_data_dir",
     "type": str,
     "default": "raw_data",
     "help": "Data dir name that stores the raw data, including x data, y data, and splits."
    },
    {"name": "x_data_dir",
     "type": str,
     "default": "x_data",
     "help": "Dir name that contains the files with features data (x data)."
    },
    {"name": "y_data_dir",
     "type": str,
     "default": "y_data",
     "help": "Dir name that contains the files with target data (y data)."
    },
    {"name": "splits_dir",
     "type": str,
     "default": "splits",
     "help": "Dir name that contains files that store split ids of the y data file."
    },
    {"name": "train_split_file",
     "default": "train_split.txt",
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., \
             'split_0_train_id', 'split_0_train_size_1024').",
    },
    {"name": "val_split_file",
     "default": "val_split.txt",
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_val_id').",
    },
    {"name": "test_split_file",
     "default": "test_split.txt",
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_test_id').",
    },
    {"name": "ml_data_outdir",  # TODO. previously ml_data_outpath
     "type": str,
     "default": "./ml_data",
     "help": "Path to save ML data (data files that can be fet to the prediction model).",
    },
    {"name": "data_format",  # TODO. rename to ml_data_format?
      "type": str,
      "default": "",
      "help": "File format to save the ML data file (e.g., '.pt', '.tfrecords')",
    },
    # {"name": "x_data_suffix",  # TODO. rename x_data_stage_fname_suffix?
    #   "type": str,
    #   "default": "data",
    #   "help": "Suffix to compose file name for storing x data (e.g., ...)."
    # },
    {"name": "y_data_suffix",  # TODO. rename y_data_stage_fname_suffix?
      "type": str,
      "default": "y_data",
      "help": "Suffix to compose file name for storing true y dataframe."
    },

]

# Parameters that are relevant to all IMPROVE training scripts
improve_train_conf = [
    {"name": "model_outdir",
     "type": str,
     "default": "./out_model", # ./models/
     "help": "Dir to save trained models.",
    },
    {"name": "model_params",  # TODO: consider renaming to "model_file_name"
     "type": str,
     "default": "model.pt",
     "help": "Filename to store trained model."
    },
    {"name": "model_file_format",  # TODO: consider making use of it
     "type": str,
     "default": ".pt",
     "help": "Suffix of the filename to store trained model."
    },
    {"name": "batch_size",
     "type": int,
     "default": 64,
     "help": "Trainig batch size."
    },
    {"name": "val_batch",
     "type": int,
     "default": 64,
     "help": "Validation batch size."
    },
    # {"name": "optimizer",
    #  "type": str,
    #  "default": "adam",
    #  "help": "Optimizer for backpropagation."
    # },
    {"name": "learning_rate",
     "type": float,
     "default": 0.0001,
     "help": "Learning rate for the optimizer."
    },
    # {"name": "loss",
    #  "type": str,
    #  "default": "mse",
    #  "help": "Loss function."
    # },
    {"name": "train_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where train data is stored."
    },
    {"name": "val_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where val data is stored."
    },
    ### TODO. probably don't need these. these will constructed from args.
    {"name": "train_data_processed",  # TODO: is this train_data.pt?
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed train data file."
    },
    {"name": "val_data_processed",  # TODO: is this val_data.pt?
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed val data file."
    },
    ###
    # {"name": "model_eval_suffix",  # TODO: what's that?
    # y_data_stage_preds_fname_suffix
    {"name": "y_data_preds_suffix",  # TODO: what's that? val_y_data_preds.csv
     "type": str,
     "default": "predicted",
     "help": "Suffix to use for name of file to store inference results."
    },
    {"name": "json_scores_suffix",
     "type": str,
     "default": "scores",
     "help": "Suffix to use for name of file to store scores."
    },
    {"name": "val_batch",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Validation batch size.",
    },
    {"name": "patience",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Iterations to wait for validation metrics getting worse before \
             stopping training.",
    },
    {"name": "early_stop_metric",
     "type": str,
     "default": "mse",
     "help": "Prediction performance metric to monitor for early stopping during \
             model training (e.g., 'mse', 'rmse').",
    },

]

# Parameters that are relevant to all IMPROVE testing scripts
improve_test_conf = [
    {"name": "test_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where test data is stored."
    },
    {"name": "test_data_processed",  # TODO: is this test_data.pt?
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed test data file."
    },
    {"name": "test_batch",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Test batch size.",
    },

]


# Combine improve configuration into additional_definitions
frm_additional_definitions = improve_basic_conf + \
    improve_preprocess_conf + \
    improve_train_conf + \
    improve_test_conf

# Required
frm_required = []


def check_path(path):
    if path.exists() == False:
        raise Exception(f"ERROR ! {path} not found.\n")


def build_paths(params):
    """ Build paths for raw_data, x_data, y_data, splits. """
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
    check_path(mainpath)

    # Raw data
    raw_data_path = mainpath / params["raw_data_dir"]
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

    # # ML data
    # ml_data_path = mainpath / params["ml_data_outdir"]
    # params["ml_data_path"] = ml_data_path
    # os.makedirs(ml_data_path, exist_ok=True)
    # check_path(ml_data_path)

    return params


def create_ml_data_outdir(params):
    """ Create a directory to store input data files for ML/DL models. """
    ml_data_outdir = Path(params["ml_data_outdir"])
    if ml_data_outdir.exists():
        print(f"ml_data_outdir already exists: {ml_data_outdir}")
    else:
        print(f"Creating ml_data_outdir: {ml_data_outdir}")
        os.makedirs(ml_data_outdir, exist_ok=True)
    check_path(ml_data_outdir)
    return ml_data_outdir


def build_ml_data_name(params: Dict, stage: str, data_format: str=""):
    """ Returns name of the ML/DL data file. E.g., train_data.pt
    TODO: params is not currently needed here. Consider removing this input arg.
    """
    data_format = "" if data_format is None else data_format
    if data_format != "" and "." not in data_format:
        data_format = "." + data_format
    # return stage + "_" + params["x_data_suffix"] + data_format
    return stage + "_" + "data" + data_format


def save_stage_ydf(ydf: pd.DataFrame, params: Dict, stage: str):
    """ Save a subset of y data samples (rows of the input dataframe).
    The "subset" refers to one of the three stages involved in developing ML
    models, including: "train", "val", or "test".

    ydf : dataframe with y data samples
    params : parameter dict
    stage (str) : "train", "val", or "test"
    """
    ydf_fname = f"{stage}_{params['y_data_suffix']}.csv"  
    ydf_fpath = Path(params["ml_data_outdir"]) / ydf_fname
    ydf.to_csv(ydf_fpath, index=False)
    return None


class ImproveBenchmark(candle.Benchmark):
    """ Benchmark for Improve Models. """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Parameters
        ----------
        required: set of required parameters for the benchmark.
        additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if frm_required is not None:
            self.required.update(set(frm_required)) # This considers global framework required arguments
        if frm_additional_definitions is not None:
            self.additional_definitions.extend(frm_additional_definitions) # This considers global framework definitions


def initialize_parameters(filepath, default_model="frm_default_model.txt", additional_definitions=None, required=None):
    """ Parse execution parameters from file or command line.

    Parameters
    ----------
    default_model : string
        File containing the default parameter definition.
    additional_definitions : List
        List of additional definitions from calling script.
    required: Set
        Required arguments from calling script.

    Returns
    -------
    gParameters: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

    # Build benchmark object
    frm = ImproveBenchmark(
        filepath=filepath,
        defmodel=default_model,
        framework="pytorch",
        prog="frm",
        desc="Framework functionality in IMPROVE",
        additional_definitions=additional_definitions,
        required=required,
    )

    gParameters = candle.finalize_parameters(frm)

    return gParameters


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

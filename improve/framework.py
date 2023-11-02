""" Basic Definitions for IMPROVE Framework. """

import os
import argparse
from pathlib import Path
from typing import List, Set, NewType, Dict, Optional # use NewType becuase TypeAlias is available from python 3.10

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
    {"name": "download",
     "type": candle.str2bool,
     "default": False,
     "help": "Flag to indicate if downloading from FTP site."
    },
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
     "default": "test_split.txt",  # TODO: what should be the default?
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_test_id').",
    },
    # {"name": "y_file",  # TODO: what's that?
    #  "type": str,
    #  "default": "y_data.tsv", # "response.tsv",
    #  "help": "File with target variable data.",
    # },
    {"name": "ml_data_outpath",
     "type": str,
     "default": "./ml_data",
     "help": "Path to save ML data (data files that can be fet to the prediction model).",
    },
    {"name": "data_suffix",
      "type": str,
      "default": "data",
      "help": "Suffix to compose file name for storing ML dataset."
    },
    {"name": "y_data_suffix",  # TODO: what's that?
      "type": str,
      "default": "y_data",
      "help": "Suffix to compose file name for storing true y dataframe."
    },

]

# Parameters that are relevant to all IMPROVE training scripts
improve_train_conf = [
    {"name": "model_outdir",
     "type": str,
     "default": "./out", # ./models/
     "help": "Path to save trained models.",
    },
    {"name": "model_params",  # TODO: consider renaming this arg into "model_file_name"
     "type": str,
     "default": "model.pt",
     "help": "Filename to store trained model."
    },
    ### TODO: added (start)
    {"name": "model_file_suffix",  # TODO: new; Should this be a model-specific arg?
     "type": str,
     "default": ".pt",
     "help": "Suffix of the filename to store trained model."
    },
    ### TODO: added (end)
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
    {"name": "train_data_processed",  # TODO: is this train_data.pt?
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed train data file."},
    {"name": "val_data_processed",  # TODO: is this val_data.pt?
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed val data file."},
    {"name": "model_eval_suffix",  # TODO: what's that?
     "type": str,
     "default": "predicted",
     "help": "Suffix to use for name of file to store inference results."},
    {"name": "json_scores_suffix",
     "type": str,
     "default": "scores",
     "help": "Suffix to use for name of file to store scores."},
    {"name": "val_batch",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Validation batch size.",
    },
    {"name": "patience",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Iterattions to wait for validation metrics getting worse before stopping training.",
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


def create_ml_data_outpath(params):
    """ ... """
    ml_data_outpath = Path(params["ml_data_outpath"])
    os.makedirs(ml_data_outpath, exist_ok=True)
    check_path(ml_data_outpath)
    return ml_data_outpath


def build_ml_data_name(params, stage, file_format: str=""):
    """ E.g., train_data.pt """
    file_format = "" if file_format is None else file_format
    return stage + "_" + params["data_suffix"] + file_format


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


# TODO: While the implementation of this func is model-specific, we may want
# to require that all models have this func defined for their models. Also,
# we need to decide where this func should be located.
def predicting(model, device, loader):
    """ Method to run predictions/inference.
    This is used in *train.py and *infer.py

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            # Is this computationally efficient?
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

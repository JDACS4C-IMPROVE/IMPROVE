"""Basic Definitions of IMPROVE Framework."""

import os
import argparse

# Check that environmental variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception(
                "ERROR ! Required system variable not specified.  You must define IMPROVE_DATA_DIR ... Exiting.\n"
            )
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

from pathlib import Path
from typing import List, Set, TypeAlias

import torch

SUPPRESS = argparse.SUPPRESS

import candle
str2bool = candle.str2bool
finalize_parameters = candle.finalize_parameters


DataPathDict: TypeAlias = dict[str, Path]


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
     "help": "Suffix to add to column name in data framework to identify predictions \
              made by the model"
    },
    {"name": "model_outdir",
     "type": str,
     "default": "./out/",
     "help": "Path to save model results.",
    },
]

# Parameters that are relevant to all IMPROVE preprocessing scripts
improve_preprocess_conf = [
    {"name": "download",
     "type": candle.str2bool,
     "default": False,
     "help": "Flag to indicate if downloading from FTP site."
    },
]

# Parameters that are relevant to all IMPROVE training scripts
improve_train_conf = [
    {"name": "y_data_suffix",
      "type": str,
      "default": "y_data",
      "help": "Suffix to compose file name for storing true y values."},
    {"name": "data_suffix",
      "type": str,
      "default": "data",
      "help": "Suffix to compose file name for storing features (x values)."},
    {"name": "model_params",
     "type": str,
     "default": "model.pt",
     "help": "Filename to store trained model parameters."},
    {"name": "train_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where train data is stored."},
    {"name": "val_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where val data is stored."},
    {"name": "train_data_processed",
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed train data file."},
    {"name": "val_data_processed",
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed val data file."},
    {"name": "model_eval_suffix",
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
    {"name": "test_data_processed",
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
        desc="Framework functionality in improve",
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


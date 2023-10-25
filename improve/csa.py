"""Functionality for Cross-Study Analysis (CSA) in IMPROVE."""

from pathlib import Path
import os
import configparser
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Tuple, Union

import improve.framework as frm

filepath = Path(__file__).resolve().parent

csa_conf = [
    {"name": "source_data",
     "nargs": "+",
     "type": str,
     "help": "List of data sources to use for training/validation.",
    },
    {"name": "target_data",
     "nargs": "+",
     "type": str,
     "help": "List of data sources to use for testing.",
    },
    {"name": "split_ids",
     "nargs": "+",
     "type": int,
     "help": "List of data samples to use for training/validation/testing.",
    },
    {"name": "model_config",
     "default": frm.SUPPRESS,
     "type": str,
     "help": "File for model configuration.",
    },
]

req_csa_args = [elem["name"] for elem in csa_conf]


class DataSplit:
    """Define structure of information for split."""
    def __init__(self, dsource: str, dtarget: str, sindex: int, tindex: int):
        self.data_source = dsource
        self.data_target = dtarget
        self.split_source_index = sindex
        self.split_target_index = tindex


def directory_tree_from_parameters(
    params: Dict,
    raw_data_check: Callable = None,
    step: str = "preprocess",
) -> Tuple[Deque, Union[frm.DataPathDict, None]]:
    """
    Check input data and construct output directory trees from parameters for cross-study analysis (CSA).

    Input and output structure depends on step.
    For preprocess step, input structure is represented by DataPathDict.
    In other steps, raw input is None and the output queue contains input and output paths.

    :param Dict params: Dictionary of parameters read
    :param Callable raw_data_check: Function that checks raw data input and returns paths to x-data/y-data/splits.
    :param string step: String to specify if this is applied during preprocess, train or test.

    :return: Paths and info about processed data output directories and raw data input.
    :rtype: (Deque, (Path, Path, Path))
    """
    inpath_dict = None
    if step == "preprocess":
        # Check that raw data is available
        inpath_dict = raw_data_check(params)
    # Create subdirectory if it does not exist
    # Structure:
    # ml_data -> {source_data-target_data} -> {split_id}
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"]) # Already checked
    outpath = mainpath / "ml_data"
    os.makedirs(outpath, exist_ok=True)
    # If used during train or test structure is slightly different
    # ml_data -> models -> {source_data-target_data} -> {split_id}
    inpath = outpath
    if step == "train": # Create structured output path
        outpath = outpath / "models"
        os.makedirs(outpath, exist_ok=True)
    elif step == "test": # Check that expected input path exists
        inpath = inpath / "models"
        if inpath.exists() == False:
            raise Exception(f"ERROR ! '{inpath}' not found.\n")
        outpath = inpath
    print("Preparing to store output under: ", outpath)

    # Create queue of cross study combinations to process and check inputs
    split_queue = deque()
    for sdata in params["source_data"]:
        for tdata in params["target_data"]:
            tag = sdata + "-" + tdata
            tagpath = outpath / tag
            inpath = inpath / tag
            if step != "preprocess" and inpath.exists() == False:
                raise Exception(f"ERROR ! '{inpath}' not found.\n")
            elif step != "test":
                os.makedirs(tagpath, exist_ok=True)

            # From this point on the depth of the path does not increase
            itagpath = inpath
            otagpath = tagpath

            if len(params["split_ids"]) == 0:
                # Need defined split ids
                raise Exception(f"ERROR ! No split ids have been defined.\n")
            else:
                for id in params["split_ids"]:
                    index = "split_" + str(id)
                    outpath = otagpath / index
                    inpath = itagpath / index
                    if step != "preprocess" and inpath.exists() == False:
                        raise Exception(f"ERROR ! '{inpath}' not found.\n")
                    elif step != "test":
                        os.makedirs(outpath, exist_ok=True)

                    tid = -1 # Used to indicate all splits
                    if sdata == tdata:
                        tid = id # Need to limit to the defined split id
                    if step == "train": # Check existence of x_data and y_data
                        for stg in ["train", "val", "test"]:
                            fname = f"{stg}_{params['y_data_suffix']}.csv"
                            ydata = inpath / fname
                            if ydata.exists() == False:
                                raise Exception(f"ERROR ! Ground truth data '{ydata}' not found.\n")
                            fname = f"{stg}_{params['data_suffix']}.pt"
                            xdata = inpath / "processed" / fname
                            if xdata.exists() == False:
                                raise Exception(f"ERROR ! Feature data '{xdata}' not found.\n")
                    elif step == "test": # Check existence of trained model
                        trmodel = inpath / params["model_params"]
                        if trmodel.exists() == False:
                            raise Exception(f"ERROR ! Trained model '{trmodel}' not found.\n")
                    split_queue.append((DataSplit(sdata, tdata, id, tid), inpath, outpath))
    return split_queue, inpath_dict


class CSAImproveBenchmark(frm.ImproveBenchmark):
    """ Benchmark for Cross-Study Analysis (CSA) Improve Models. """

    def read_config_file(self, file: str):
        """
        Functionality to read the configue file specific for each
        benchmark.

        :param string file: path to the configuration file

        :return: parameters read from configuration file
        :rtype: ConfigDict
        """

        # Read CSA workflow configuration
        fileParams = super().read_config_file(file)

        # Read model configuration
        confmodelfile = fileParams["model_config"]
        fileparams_model = super().read_config_file(confmodelfile)

        # Combine both specifications
        fileParams.update(fileparams_model)

        return fileParams



def initialize_parameters(filepath, default_model="csa_default_model.txt", additional_definitions=None, required=None, topop=None):
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
    if required is not None:
        req_csa_args.extend(required)

    # Build benchmark object
    csabmk = CSAImproveBenchmark(
        filepath=filepath,
        defmodel=default_model,
        framework="pytorch",
        prog="csa",
        desc="CSA workflow functionality in improve",
        additional_definitions=additional_definitions + csa_conf,
        required=req_csa_args,
    )

    gParameters = frm.finalize_parameters(csabmk)
    if topop is not None:
        for k in topop:
            gParameters.pop(k)

    return gParameters

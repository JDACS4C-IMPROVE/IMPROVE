""" 
This module includes functionality for file and directory management, including path validation, directory creation, and file format handling.
"""
import os
from pathlib import Path
import json
from typing import Dict, List, Union


# def build_ml_data_name(params: Dict, stage: str, file_format: str=""):
def build_ml_data_file_name(data_format: str, stage: str) -> str:
    """Return the name of the ML/DL data file.

    Args:
        data_format (str): The format of the data file (e.g., '.pt').
        stage (str): The stage of the data (e.g., 'train', 'val', 'test').

    Returns:
        str: The constructed file name for the ML/DL data file.
    """
    data_file_format = get_file_format(file_format=data_format)
    ml_data_file_name = stage + "_" + "data" + data_file_format
    return ml_data_file_name


def build_model_path(model_file_name: str,
                     model_file_format: str,
                     model_dir: Union[Path, str]) -> Path:
    """Build the path to save the trained model.

    Args:
        model_file_name (str): The name of the model file.
        model_file_format (str): The format of the model file (e.g., '.pt').
        model_dir (Union[Path, str]): The directory path to save the model.

    Returns:
        Path: The constructed model file path.
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


def build_paths(params: Dict) -> Dict:
    """Build paths for raw_data, x_data, y_data, and splits.
    These paths determine directories for a benchmark dataset.
    TODO: consider renaming to build_benchmark_data_paths()

    Args:
        params (Dict): Dict of parameters containing directory information (IMPROVE/CANDLE params).

    Returns:
        Dict: Updated dictionary with built paths (IMPROVE/CANDLE params).
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


def check_path(path: Union[Path, str]):
    """Check if a given path exists.

    Args:
        path (Path): The path to check.

    Raises:
        Exception: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise Exception(f"ERROR ! {path} not found.\n")


def create_outdir(outdir: Union[Path, str]) -> Path:
    """Create a directory if it does not already exist.

    Args:
        outdir (Union[Path, str]): The directory path to create.

    Returns:
        Path: The created or existing directory path.
    """
    outdir = Path(outdir)
    if outdir.exists():
        print(f"Dir already exists: {outdir}")
    else:
        print(f"Creating dir: {outdir}")
        os.makedirs(outdir, exist_ok=True)
    check_path(outdir)
    return outdir


def get_file_format(file_format: Union[str, None] = None) -> str:
    """Clean and standardize the file format string.

    Args:
        file_format (Union[str, None]): The file format string.

    Returns:
        str: The standardized file format, prefixed with a dot if necessary.

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



# ----------------------------------------------------------------------
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
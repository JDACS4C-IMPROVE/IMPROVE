""" Basic definitions for IMPROVE framework. """

import argparse
import json
import os
import time
from pathlib import Path
from ast import literal_eval
# use NewType becuase TypeAlias is available from python 3.10
from typing import List, Set, Union, NewType, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .utils_app_generic import (
    _determine_impute, 
    _determine_scale, 
    _determine_subset, 
    _impute_features, 
    _scale_features, 
    _subset_features, 
    _get_full_input_path, 
    _get_full_preprocess_path, 
    _get_all_splits, 
    _get_stage_splits
)


def get_y_data(split_file, benchmark_dir, y_data_file, split_id='split_id', sep='\t'):
    """Gets y data for a given split file.

    Args:
        split_file (Union[str, Path, list of str, list of Path]): Name of split file if in benchmark data, otherwise path to split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        y_data_file (str): Name of y data file.
        split_id (str): Name of column containing the split ID (default: 'split_id').
        sep (str): Separator for y data file (default: '\t').

    Returns:
        pd.DataFrame: Y data dataframe for given split.
    """
    # get path to y_data file, read data
    response_path = _get_full_input_path(y_data_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    if split_id not in df.columns:
        df = df.reset_index()
    # get path to splits file, read data
    ids = _get_stage_splits(split_file, benchmark_dir)
    # subset y_data based on split given
    df = df[df[split_id].isin(ids)]
    return df


def get_all_response_data(train_split_file, val_split_file, test_split_file, benchmark_dir, response_file, sep='\t'):
    """Gets response data for all given split file. Denotes stage of split in col 'split' with 'train', 'val', or 'test'.

    Args:
        train_split_file (Union[str, Path, list of str, list of Path]): Name of train split file if in benchmark data, otherwise path to train split file. Can be a list of str or Path.
        val_split_file (Union[str, Path, list of str, list of Path]): Name of val split file if in benchmark data, otherwise path to val split file. Can be a list of str or Path.
        test_split_file (Union[str, Path, list of str, list of Path]): Name of test split file if in benchmark data, otherwise path to test split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        response_file (str): Name of response file
        sep (str): Separator for response file (default: '\t').

    Returns:
        pd.DataFrame: Response dataframe for all splits with col 'split' denoting split type ('train', 'val', or 'test').
    """
    # get path to y_data file, read data
    response_path = _get_full_input_path(response_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    # get path to splits files, read data
    train, val, test = _get_all_splits(train_split_file, val_split_file, test_split_file, benchmark_dir)
    # label y_data with split column, populated with the appropriate stage name
    df['split'] = "NA"
    df.loc[train, 'split'] = "train"
    df.loc[val, 'split'] = "val"
    df.loc[test, 'split'] = "test"
    # drop y_data not in any stage
    df = df[df['split'].notna()]
    return df



def get_x_data(file, benchmark_dir, column_name, dtype=None):
    """Generic function to get x data. Sets index to ID. Sets dtype if specified.

    Args:
        file (Union[str, Path]): Name of x data file if in benchmark data, otherwise path to x data file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        column_name (str): Name of ID column for x data.
        dtype (str): dtype to enforce for this x data.

    Returns:
        pd.DataFrame: x data (with dtype if specified), index set to ID.
    """
    file_path = _get_full_input_path(file, benchmark_dir, 'x_data')
    data = pd.read_csv(file_path, sep='\t')
    # enforce index and type
    data.set_index(column_name, inplace=True)
    if dtype is not None:
        data = data.astype(dtype)
    return data

def get_y_data_with_features(y_data_df, feature_df, column_name):
    """Takes a y data DataFrame and feature DataFrame(s) and returns a y data DataFrame
    that contains only rows that have available features for the feature type(s) provided. 
    All features in the list must have the same ID type (e.g. drug or cell). If a list is given, 
    only rows will be retained if all features in the list are available.

    Args:
        y_data_df (pd.DataFrame): Response DataFrame.
        feature_df (pd.DataFrame or List of pd.DataFrame): Feature DataFrame or a list of feature DataFrames of the same ID (drug or cell). ID must be index, as with all improvelib functions.
        column_name (str): Name of ID column for x data.
    
    Returns:
        pd.DataFrame: Y data DataFrame containing only the rows with features available.
    """

    if isinstance(feature_df, list):
        for df in feature_df:
            intersect_list = list(set(df.index.tolist()) & set(y_data_df[column_name]))
            y_data_df = y_data_df[y_data_df[column_name].isin(intersect_list)]
    else:
        intersect_list = list(set(feature_df.index.tolist()) & set(y_data_df[column_name]))
        y_data_df = y_data_df[y_data_df[column_name].isin(intersect_list)]
    return y_data_df

def get_features_in_y_data(feature_df, y_data_df, column_name):
    """Takes a feature DataFrame and a y data DataFame and returns the feature DataFrame that 
    contains only features that are present in the given y data DataFrame.

    Args:
        feature_df (pd.DataFrame): Feature DataFrame. ID must be index, as with all improvelib functions.
        y_data_df (pd.DataFrame): Y data DataFrame.
        column_name (str): Name of ID column for x data.

    Returns:
        pd.DataFrame: Feature DataFrame containing only the rows with features that are used in the y data.

    """
    intersect_list = list(set(feature_df.index.tolist()) & set(y_data_df[column_name]))
    feature_df = feature_df[feature_df.index.isin(intersect_list)]
    return feature_df


def determine_transform(x_data_df, x_data_name, x_transform_list, output_dir):
    """
    Sets the transformations (imputations, scaling, and/or subsetting) features based on a list of lists of [[strategy, subtype]]. 
    Transformation values are determined by the training set.
    Saves a dictionary containing the details needed to perform the specified transformations on all sets.
    Before using this function...

    Args:
        x_data_df (pd.DataFrame): The input DataFrame, column names must be Entrez IDs, index must be IDs.
        x_data_name (str): Name for the saved tranformation dictionary (.json will be added). 
        x_transform_list (List): List of lists of [[strategy, subtype]], e.g. [['subset', 'L1000_SYMBOL'], ['scale', 'StandardScaler']].
        output_dir: Should be set to params['output_dir'].
    """
    # NEED TO EITHER: limit to one of each, or enforce some sort of limit / order
    transform_dict = {}
    if (x_transform_list != []) and (x_transform_list != None) and (x_transform_list != 'None'):
        if isinstance(x_transform_list, str):
            x_transform_list = literal_eval(x_transform_list)
        for n in x_transform_list:
            if not len(n) == 2:
                print(f"Each transformation list must have two items. Skipping {n}.")
            else:
                strategy = n[0]
                subtype = n[1]
                if strategy not in ['impute','scale', 'subset']:
                    print(f"{strategy} is an invalid strategy. Choose 'impute', 'scale', or 'subset'. Skipping {n}.")
                elif strategy == 'impute':
                    print(f"Determining {strategy} with {subtype}.")
                    impute_value, x_data_df = _determine_impute(x_data_df, subtype)
                    transform_dict['impute'] = impute_value
                elif strategy == 'scale':
                    print(f"Determining {strategy} with {subtype}.")
                    scaler_path = str(os.path.join(output_dir, x_data_name))
                    scaler_name, x_data_df = _determine_scale(x_data_df, subtype, scaler_path)
                    transform_dict['scale'] = scaler_name
                elif strategy == 'subset':
                    print(f"Determining {strategy} with {subtype}.")
                    subset_list, x_data_df = _determine_subset(x_data_df, subtype)
                    transform_dict['subset'] = subset_list
    transform_name = os.path.join(output_dir, x_data_name + '.json')
    with open(transform_name, 'w') as f:
        json.dump(transform_dict, f, indent=4)
    # but what if we need to transform multiple x data?

def transform_data(df, transform_file_name, preprocess_dir):
    """
    Transforms (imputes, scales, and/or subsets) features based the transformations determined on the training set with determine_transform(). 
    Reads the saved dictionary containing the details needed to perform the specified transformations on all sets, and performs the 
    transformations on the given data.

    Args:
        df (pd.DataFrame): The input feature DataFrame, column names must be feature IDs (e.g. gene names), index must be IDs (e.g. cell line names).
        transform_file_name (str): Name of the file name used in determine_transform().
        preprocess_dir (str): Should be params['output_dir'].

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """

    # open dictionary
    # add a check that this doesn't already contain a .json
    transform_file_name = transform_file_name + '.json'
    transform_dict_path = _get_full_preprocess_path(transform_file_name, preprocess_dir)
    with open(transform_dict_path, 'r') as f:
        transform_dict = json.load(f)
    for key, value in transform_dict.items():
        if key == 'impute':
            print(f"Imputing features with {value}.")
            df = _impute_features(df, value)
        elif key == 'scale':
            print(f"Scaling features with {value}.")
            df = _scale_features(df, value)
        elif key == 'subset':
            print(f"Subsetting features with {value}.")
            df = _subset_features(df, value)
        else:
            print(f"Invalid tranformation type {key}. Must be 'impute', 'scale', or 'subset'.")
    return df

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
                assert np.array_equal(v1, v2, equal_nan=True), "Loaded y data\
                    array is not equal to the true array"

        df = pd.concat([rsp_df, pred_df], axis=1)

    else:
        df = pred_df.copy()

    df.to_csv(ydf_out_fpath, index=False)  # Save predictions df

    return None

def compute_performance_scores(y_true: np.array,
                               y_pred: np.array,
                               stage: str, 
                               metric_type: str, 
                               output_dir: str,
                               y_prob=None):
    """Evaluate predictions according to specified metrics.

    Metrics are evaluated. Scores are stored in specified path and returned.


    :params array y_true: Array with ground truth values.
    :params array y_pred: Array with model predictions.
    :params str stage: String specified if evaluation is with respect to
            validation or testing set.
    :params str metric_type: Either classification or regression.
    :params str output_dir: Directory to write results.
    :params: array y_prob: Array with target scores from classification model, defaults to None.

    :return: Python dictionary with metrics evaluated and corresponding scores.
    :rtype: dict
    """
    # Compute multiple performance scores
    scores = compute_metrics(y_true, y_pred, metric_type, y_prob)

    # Add val_loss metric
    #key = f"{stage}_loss"
    #scores[key] = scores[params["loss"]]

    scores_fname = f"{stage}_scores.json"
    scorespath = Path(output_dir) / scores_fname

    with open(scorespath, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    # TODO. do we still need to print IMPROVE_RESULT?
    if metric_type == "regression":
        if stage == "val":
            print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
            print("Validation scores:\n\t{}".format(scores))
        elif stage == "test":
            print("Inference scores:\n\t{}".format(scores))
        else:
            print("Invalid stage: must be 'val' or 'test'.")
    elif metric_type == "classification":
        if stage == "val":
            print("Validation scores:\n\t{}".format(scores))
        elif stage == "test":
            print("Inference scores:\n\t{}".format(scores))
        else:
            print("Invalid stage: must be 'val' or 'test'.")
    else: 
        print("Invalid metric_type provided. Choose 'classification' or 'regression'.")

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

def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Search for common data in a reference column and retain only those rows.

    Args:
        df1 (pd.DataFrame): First dataframe.
        df2 (pd.DataFrame): Second dataframe.
        ref_col (str): The reference column to find the common values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of DataFrames after filtering for common data.
    """
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def get_common_elements(list1: List, list2: List, verbose: bool = False) -> List:
    """Return a list of elements that the provided lists have in common.

    Args:
        list1 (List): One list.
        list2 (List): Another list.
        verbose (bool): Flag for verbosity. If True, info about computations is displayed. Default is False.

    Returns:
        List: List of common elements.
    """
    in_common = list(set(list1).intersection(set(list2)))
    if verbose:
        print("Elements in common count: ", len(in_common))
    return in_common

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
        if not hasattr(self, 'time_diff_dict'):
            self.timer_end()
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
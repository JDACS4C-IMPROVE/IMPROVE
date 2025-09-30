# Standard library imports
from ast import literal_eval
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np

import joblib
import json
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from improvelib.statics import L1000_ENTREZ, L1000_SYMBOL, LINCS_SYMBOL


# requires ID to be index

def _determine_impute(df, subtype):
    """
    Determines the imputation based on the training set with the specified imputation type. 

    Args:
        df (pd.DataFrame): The input feature DataFrame, column names must be feature IDs (e.g. gene names), index must be IDs (e.g. cell line names).
        subtype (str): The type of imputation to be performed.

    Returns:
        impute_value (numeric or list of numerics): The imputation value(s).
        df (pd.DataFrame): The df with NaN values imputed by the given impute_value(s).
    """
    # add check that it's only numerical
    if subtype == 'zero':
        impute_value = 0
    elif subtype == 'mean':
        impute_value = df.mean(axis=None)
    elif subtype == 'mean_col':
        impute_value = df.mean()
    elif subtype == 'median':
        impute_value = df.median(axis=None)
    elif subtype == 'median_col':
        impute_value = df.median()
    else:
        print(f"The specified imputation ({subtype}) is not implemented.")
    df = _impute_features(df, impute_value)
    return impute_value, df

def _determine_scale(df, subtype, data_name):
    """
    Determines the scaler based on the training set with the specified scaler type. Uses scikit-learn scalers.

    Args:
        df (pd.DataFrame): The input feature DataFrame, column names must be feature IDs (e.g. gene names), index must be IDs (e.g. cell line names).
        subtype (str): The type of scaler to be used.
        data_name (str): String representing data being scaled, used to save the scaler.

    Returns:
        scaler_name (str): The name of the saved scaler (data_name + '_scaler.gz').
        df (pd.DataFrame): The scaled df with the given scaler type.
    """
    # add check that's it's only numerical
    # determine scaler to use
    if subtype == 'std' or subtype == 'StandardScaler':
        scaler = StandardScaler()
    elif subtype == 'minmax' or subtype == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif subtype == "minabs" or subtype == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif subtype == "robust" or subtype == 'RobustScaler':
        scaler = RobustScaler()
    elif subtype == None or subtype == 'None':
        scaler = None
    else:
        print(f"The specified scaler ({subtype}) is not implemented (no df scaling).")
        scaler = None
    # fit scaler on train data
    if scaler is None:
        fit_scaler = None
    else:
        fit_scaler = scaler.fit(df)
    scaler_name = data_name + '_scaler.gz'
    joblib.dump(fit_scaler, scaler_name)
    df = _scale_features(df, scaler_name)
    return scaler_name, df


def _determine_subset(df, subtype):
    """
    Determines the feature subset based on the training set with the specified subset type. 

    Args:
        df (pd.DataFrame): The input feature DataFrame, column names must be feature IDs (e.g. gene names), index must be IDs (e.g. cell line names).
        subtype (str): The type of subset to be used. Can be a path to a plain text file.

    Returns:
        subset_list (list of str): List of the feature IDs to be used.
        df (pd.DataFrame): The subsetted df with the given subset type.
    """
    # add check that it's only numerical
    if subtype == 'L1000_SYMBOL':
        # need id type and check here
        df_columns = set(df.columns.to_list())
        subset_list = [item for item in L1000_SYMBOL if item in df_columns]
    elif subtype == 'L1000_ENTREZ':
        # need id type and check here
        df_columns = set(df.columns.to_list())
        subset_list = [item for item in L1000_ENTREZ if item in df_columns]
    elif subtype == 'LINCS_SYMBOL':
        # need id type and check here
        df_columns = set(df.columns.to_list())
        subset_list = [item for item in LINCS_SYMBOL if item in df_columns]
        #subset_list = list(set(LINCS_SYMBOL) & set(df.columns.to_list())) # this does not preserve order!
    elif subtype == 'high_variance':
        vars = df.var()
        var_threshold = 0.8
        vars_subset = vars[vars < var_threshold]
        subset_list = vars_subset.columns.tolist()
    elif os.path.isfile(subtype):
        try:
            loaded_list = list(np.loadtxt(subtype, dtype=str))
            df_columns = set(df.columns.to_list())
            subset_list = [item for item in loaded_list if item in df_columns]
        except:
            print(f"There was an error trying to use {subtype} to subset the data. Ensure the file is a plain text list of gene IDs, with each ID on a new line. \n Skipping subset with {subtype}.")
 
    else:
        print(f"The specified subset ({subtype}) is not implemented.")
    df = _subset_features(df, subset_list)
    return subset_list, df



def _impute_features(df, impute_value):
    # type checking should be somewhere
    df = df.fillna(impute_value)
    return df

def _scale_features(df, scaler_path):
    # path is going to be output dir, deal with that here or in above function
    scaler = joblib.load(scaler_path)
    arr = scaler.transform(df)
    df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    return df

def _subset_features(df, feature_list):
    # assumes all column names in list are actually in the df
    df = df[feature_list]
    return df

# need a function to record feature order



###########################################
######### I/O FUNCTIONS ###################
###########################################


def _get_full_input_path(fname, benchmark_dir, benchmark_type) -> None:
    """Check if a name is a full path, if not check if it is in benchmark dir

    Args:
        fname (Union[str, Path]): Name of file or path to check.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        benchmark_type (str): one of ['x_data', 'y_data', 'splits'].

    Returns:
      Path: Path to fname.

    Raises:
        Exception: If the path does not exist.
    """
    if not os.path.isfile(fname):
        # if it is not a full path, check if it's in benchmarks
        fname = os.path.join(benchmark_dir, benchmark_type, fname)
        # if it is not a full path, or in benchmarks, raise an error
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"ERROR! {fname} not found.\n")
    return fname

def _get_full_preprocess_path(fname, preprocess_dir) -> None:
    """Check if a name is a full path, if not check if it is in the preprocess (output) directory.

    Args:
        fname (Union[str, Path]): Name of file or path to check.
        preprocesss_dir (Union[str, Path]): Path to directory where preprocess output data is stored.

    Returns:
      Path: Path to fname.

    Raises:
        Exception: If the path does not exist.
    """
    if not os.path.isfile(fname):
        # if it is not a full path, check if it's in benchmarks
        fname = os.path.join(preprocess_dir, fname)
        # if it is not a full path, or in benchmarks, raise an error
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"ERROR! {fname} not found.\n")
    return fname


def _get_all_splits(train_split_file, val_split_file, test_split_file, benchmark_dir):
    """Gets split indexes for train, val, and test. Split files can be a single file or a list of files, but the lengths of the lists must match.

    Args:
        train_split_file (Union[str, Path, list of str, list of Path]): Name of train split file if in benchmark data, otherwise path to train split file. Can be a list of str or Path.
        val_split_file (Union[str, Path, list of str, list of Path]): Name of val split file if in benchmark data, otherwise path to val split file. Can be a list of str or Path.
        test_split_file (Union[str, Path, list of str, list of Path]): Name of test split file if in benchmark data, otherwise path to test split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.

    Returns:
        list: Split indexes for train split.
        list: Split indexes for val split.
        list: Split indexes for test split.

    Raises:
        Exception: If the splits are a mix of lists and strings.
    """
    try:
        train_split_file = literal_eval(train_split_file)
        val_split_file = literal_eval(val_split_file)
        test_split_file = literal_eval(test_split_file)
    except Exception:
        pass 
    if isinstance(train_split_file, str) and isinstance(val_split_file, str) and isinstance(test_split_file, str):
        # get path to splits files, read data
        train_split_path = _get_full_input_path(train_split_file, benchmark_dir, 'splits')
        val_split_path = _get_full_input_path(val_split_file, benchmark_dir, 'splits')
        test_split_path = _get_full_input_path(test_split_file, benchmark_dir, 'splits')
        train = list(np.loadtxt(train_split_path,dtype=int))
        val = list(np.loadtxt(val_split_path,dtype=int))
        test = list(np.loadtxt(test_split_path,dtype=int))
    elif isinstance(train_split_file, list) and isinstance(val_split_file, list) and isinstance(test_split_file, list):
        if not (len(train_split_file) == len(val_split_file) == len(test_split_file)):
            print("WARNING! 'train_split_file', 'val_split_file', and 'test_split_file' are lists, but not of the same length.\n")
        train = _get_stage_splits(train_split_file, benchmark_dir)
        val = _get_stage_splits(val_split_file, benchmark_dir)
        test = _get_stage_splits(test_split_file, benchmark_dir)
    else:
        raise TypeError("'train_split_file', 'val_split_file', and 'test_split_file' are a mix of lists and strings. Exiting.")
    return train, val, test

def _get_stage_splits(split_file, benchmark_dir):
    """Gets split indexes for a single stage. Can be a single file or a list of files.

    Args:
        split_file (Union[str, Path, list of str, list of Path]): Name of split file if in benchmark data, otherwise path to split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.

    Returns:
        list: Split indexes for split.

    Raises:
        Exception: If the splits file is not a string or a list.
    """
    try:
        split_file = literal_eval(split_file)
    except Exception:
        pass 
    if isinstance(split_file, str):
        # get path to splits files, read data
        split_path = _get_full_input_path(split_file, benchmark_dir, 'splits')
        splits = list(np.loadtxt(split_path,dtype=int))
    elif isinstance(split_file, list):
        splits = []
        for m in range(len(split_file)):
            split_path = _get_full_input_path(split_file[m], benchmark_dir, 'splits')
            splits = splits + list(np.loadtxt(split_path,dtype=int))
    else:
        raise TypeError(f"Split file {split_file} is not a string or a list. Exiting.")
    return splits


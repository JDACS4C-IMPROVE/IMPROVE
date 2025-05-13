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

def get_response_data(split_file, benchmark_dir, response_file, sep='\t'):
    """Gets response data for a given split file.

    Args:
        split_file (Union[str, Path, list of str, list of Path]): Name of split file if in benchmark data, otherwise path to split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        response_file (str): Name of response file.
        sep (str): Separator for response file (default: '\t').

    Returns:
        pd.DataFrame: Response dataframe for given split.
    """
    # get path to y_data file, read data
    response_path = _get_full_input_path(response_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    # get path to splits file, read data
    ids = _get_stage_splits(split_file, benchmark_dir)
    # subset y_data based on split given
    df = df.loc[ids]
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



def get_x_data(file, benchmark_dir, column_name, dtype=None, gene_id=None):
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

def get_response_with_features(response_df, feature_df, column_name):
    if isinstance(feature_df, list):
        for df in feature_df:
            intersect_list = list(set(df.index.tolist()) & set(response_df[column_name]))
            response_df = response_df[response_df[column_name].isin(intersect_list)]
    else:
        intersect_list = list(set(feature_df.index.tolist()) & set(response_df[column_name]))
        response_df = response_df[response_df[column_name].isin(intersect_list)]
    return response_df

def get_features_in_response(feature_df, response_df, column_name):
    intersect_list = list(set(feature_df.index.tolist()) & set(response_df[column_name]))
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
        x_transform_list (str): List of lists of [[strategy, subtype]], e.g. [['subset', 'L1000_SYMBOL'], ['scale', 'StandardScaler']].
        output_dir: Should be set to params['output_dir'].
    """
    # NEED TO EITHER: limit to one of each, or enforce some sort of limit / order

    # need to loop through the tranformation list
    # need to save all in a dictionary
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
                    scaler_name, x_data_df = _determine_scale(x_data_df, subtype, x_data_name)
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


###########################################
######### X DATA TRANSFORMATIONS ##########
###########################################

# requires ID to be index

def _determine_impute(df, subtype):
    # add check that it's only numerical
    if subtype == 'zero':
        impute_value = 0
    if subtype == 'mean':
        impute_value = df.mean(axis=None)
    if subtype == 'mean_col':
        impute_value = df.mean()
    else:
        print(f"The specified imputation ({subtype}) is not implemented.")
    df = _impute_features(df, impute_value)
    return impute_value, df

def _determine_scale(df, subtype, data_name):
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
    # add check that it's only numerical
    if subtype == 'L1000_SYMBOL':
        # need id type and check here
        subset_list = list(set(L1000_SYMBOL) & set(df.columns.to_list()))
    elif subtype == 'L1000_ENTREZ':
        # need id type and check here
        subset_list = list(set(L1000_ENTREZ) & set(df.columns.to_list()))
    elif subtype == 'LINCS_SYMBOL':
        # need id type and check here
        subset_list = list(set(LINCS_SYMBOL) & set(df.columns.to_list()))
    elif subtype == 'high_variance':
        vars = df.var()
        var_threshold = 0.8
        vars_subset = vars[vars < var_threshold]
        subset_list = vars_subset.columns.tolist()
    elif os.path.isfile(subtype):
        try:
            loaded_list = list(np.loadtxt(subtype, dtype=str))
            subset_list = list(set(loaded_list) & set(df.columns.to_list()))
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
    df = scaler.transform(df)
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


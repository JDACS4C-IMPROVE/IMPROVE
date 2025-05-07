# Standard library imports
from ast import literal_eval
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np

from improvelib.statics import L1000_ENTREZ, L1000_SYMBOL


def _get_x_data(file, benchmark_dir, column_name, norm, dtype, gene_id=None):
    """Generic function to get x data. Sets index to ID and checks dtype.

    Args:
        file (Union[str, Path]): Name of x data file if in benchmark data, otherwise path to x data file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        column_name (str): Name of ID column for x data.
        norm (list): Normalization to perform on this x data.
        dtype (str): dtype to enforce for this x data.

    Returns:
        pd.DataFrame: x data (with normalization if specified), index set to ID.
    """
    file_path = _get_full_input_path(file, benchmark_dir, 'x_data')
    data = pd.read_csv(file_path, sep='\t')
    # enforce index and type
    data.set_index(column_name, inplace=True)
    if (dtype is not None) and (dtype is not 'None'):
        data = data.astype(dtype)
    # call normalization if needed
    data = _transform_cell_features(data, norm, gene_id)
    return data


###########################################
######### X DATA TRANSFORMATIONS ##########
###########################################

# requires ID to be index

def _transform_cell_features(df, norm_list, gene_id):
    """Transforms (subsets and/or normalizes) cell features based on a list of lists of [[strategy, subtype]]. Transformations are performed
    in the order listed. For example, [['subset', 'L1000'], ['normalize', 'zscale']] will first subset the columns to genes in LINCS1000, and
    then normalize the remaining data by Z-scaling.

    Args:
        df (pd.DataFrame): The input DataFrame, column names must be Entrez IDs, index must be IDs.
        subtype (str): List of lists of [[strategy, subtype]], e.g. [['subset', 'L1000'], ['normalize', 'zscale']].

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    norm_df = df
    if (norm_list != []) and (norm_list != None) and (norm_list != 'None'):
        if isinstance(norm_list, str):
            print("norm_list is a string. Converting to list.")
            norm_list = literal_eval(norm_list)
        for n in norm_list:
            if not len(n) == 2:
                print(f"Each processing list must have two items. Skipping {n}.")
            else:
                strategy = n[0]
                subtype = n[1]
                if strategy not in ['normalize', 'subset']:
                    print(f"{strategy} is an invalid strategy. Choose 'normalize' or 'subset'. Skipping {n}.")
                elif strategy == 'normalize':
                    print(f"Running {strategy} with {subtype}.")
                    norm_df = _normalize_features(df, subtype)
                elif strategy == 'subset':
                    print(f"Running {strategy} with {subtype}.")
                    norm_df = _subset_features(df, subtype, gene_id)
    return norm_df

def _normalize_features(df, subtype):
    """Normalizes columns in a Pandas DataFrame based on the specified subtype. Currently only Z-scaling is implemented.

    Args:
        df (pd.DataFrame): The input DataFrame, column names must be Entrez IDs, index must be IDs.
        subtype (str): Currently must be 'zscale', other normalizations are not yet supported.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    norm_df = df
    if subtype == 'zscale':
        norm_df = _z_scale_dataframe(df.copy())
    else:
        print("zscale is the only implemented normalization")
    return norm_df

def _subset_features(df, subtype, gene_id):
    """Subsets columns in a Pandas DataFrame based on the specified subtype.

    Args:
        df (pd.DataFrame): The input DataFrame, column names must be Entrez IDs, index must be IDs.
        subtype (str): Either 'high_variance', 'L1000', or a path to a plain text list of ENTREZ gene IDs, with each ID on a new line.

    Returns:
        pd.DataFrame: The subsetted DataFrame.
    """
    norm_df = df
    if subtype == 'high_variance':
        norm_df = _subset_high_variance(df)
    elif subtype == 'L1000':
        norm_df = _subset_L1000(df, gene_id)
    elif os.path.isfile(subtype):
        try:
            sub_list = list(np.loadtxt(subtype, dtype=str))
            inter_list = list(set(sub_list) & set(df.columns.to_list()))
            norm_df = df[inter_list]
        except:
            print(f"There was an error trying to use {subtype} to subset the data. Ensure the file is a plain text list of ENTREZ gene IDs, with each ID on a new line. \n Skipping subset with {subtype}.")
    else:
        print(f"Subset with {subtype} is invalid. Please choose 'high_variance', 'L1000', or provide the path to a file with ENTREZ IDs.")
    return norm_df

def _subset_L1000(df, gene_id):
    """Subsets columns in a Pandas DataFrame to only those with a column name in LINCS1000 genes list.

    Args:
        df (pd.DataFrame): The input DataFrame, column names must be Entrez IDs, index must be IDs.

    Returns:
        pd.DataFrame: The subsetted DataFrame with only LINCS1000 genes.
    """
    # FUTURE: this could take a parameter for any subset
    if gene_id == 'Entrez':
        inter_list = list(set(L1000_ENTREZ) & set(df.columns.to_list()))
    elif gene_id == 'Symbol':
        inter_list = list(set(L1000_SYMBOL) & set(df.columns.to_list()))
    else:
        raise ValueError(f"ERROR! Gene ID type provided was {gene_id} but must be either 'Entrez' or 'Symbol'.\n")
    sub_df = df[inter_list]
    return sub_df

def _subset_high_variance(df, var_threshold=0.8):
    """Subsets columns in a Pandas DataFrame to only those with variance greater than the specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame, index must be IDs.
        var_threshold (float): Threshold of variance to subset by if variance is greater than.

    Returns:
        pd.DataFrame: The subsetted DataFrame.
    """
    vars = df.var()
    vars_subset = vars[vars < var_threshold]
    df_subset = df[vars_subset.index]
    return df_subset

def _z_scale_dataframe(df):
    """Z-scales all columns in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame, index must be IDs.

    Returns:
        pd.DataFrame: The z-scaled DataFrame.
    """
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

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
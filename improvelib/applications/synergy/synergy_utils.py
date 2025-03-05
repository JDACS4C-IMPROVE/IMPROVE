"""
This module provides utilities for loading and processing response data for 
synergy models in the IMPROVE framework. It also provides 
functionality to filter dataframes, retaining only the common IDs shared between them.
"""

# Standard library imports
from ast import literal_eval
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np

from improvelib.applications.synergy.synergy_statics import L1000_ENTREZ, L1000_SYMBOL

def z_scale_dataframe(df):
  """
  Z-scales all columns in a Pandas DataFrame.

  Args:
      df (pd.DataFrame): The input DataFrame.

  Returns:
      pd.DataFrame: The z-scaled DataFrame.
  """
  for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()
  return df

def get_full_input_path(fname, benchmark_dir, benchmark_type) -> None:
    """Check if a name is a full path, if not check if it is in benchmark dir

    Args:
        fpath (Union[str, Path]): Path to check.
        benchmark_type: one of ['x_data', 'y_data', 'splits']

    Raises:
        Exception: If the path does not exist.
    """
    if not os.path.isfile(fname):
        # if it is not a full path, check if it's in benchmarks
        fname = os.path.join(benchmark_dir, benchmark_type, fname)
        # if it is not a full path, or in benchmarks, raise an error
        if not os.path.isfile(fname):
            raise Exception(f"ERROR! {fname} not found.\n")
    return fname


##### Y_DATA FUNCTIONS #####
def get_response_data(split_file, benchmark_dir, response_file='synergy.tsv', sep='\t'):
    # get path to y_data file, read data
    response_path = get_full_input_path(response_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    # ensures the first four columns (cell ID, drug ID, drug ID, study) are strings
    df[df.columns[0:4]] = df[df.columns[0:4]].astype(str)
    # ensures the rest of the columns are floats
    df[df.columns[4:]] = df[df.columns[4:]].astype(float)
    # get path to splits file, read data
    split_path = get_full_input_path(split_file, benchmark_dir, 'splits')
    ids = pd.read_csv(split_path, header=None)[0].tolist()
    # subset y_data based on split given
    df = df.loc[ids]
    return df

def get_all_response_data(train_split_file, val_split_file, test_split_file, benchmark_dir, response_file='synergy.tsv', sep='\t'):
    # get path to y_data file, read data
    response_path = get_full_input_path(response_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    # ensures the first four columns (cell ID, drug ID, drug ID, study) are strings
    df[df.columns[0:4]] = df[df.columns[0:4]].astype(str)
    # ensures the rest of the columns are floats
    df[df.columns[4:]] = df[df.columns[4:]].astype(float)
    # get path to splits files, read data
    train_split_path = get_full_input_path(train_split_file, benchmark_dir, 'splits')
    val_split_path = get_full_input_path(val_split_file, benchmark_dir, 'splits')
    test_split_path = get_full_input_path(test_split_file, benchmark_dir, 'splits')
    train = list(np.loadtxt(train_split_path,dtype=int))
    val = list(np.loadtxt(val_split_path,dtype=int))
    test = list(np.loadtxt(test_split_path,dtype=int))
    # label y_data with split column, populated with the appropriate stage name
    df['split'] = "NA"
    df.loc[train, 'split'] = "train"
    df.loc[val, 'split'] = "val"
    df.loc[test, 'split'] = "test"
    # drop y_data not in any stage
    df = df[df['split'].notna()]
    return df

#### X_DATA FUNCTIONS
def get_x_data(file, benchmark_dir, column_name, norm, dtype):
    file_path = get_full_input_path(file, benchmark_dir, 'x_data')
    data = pd.read_csv(file_path, sep='\t')
    # enforce index and type
    data.set_index(column_name, inplace=True)
    data = data.astype(dtype)
    # call normalization if needed
    data = normalize_cell_features(data, norm)
    return data

def get_cell_transcriptomics(file, benchmark_dir, cell_column_name, norm):
    data = get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64')
    return data

def get_cell_cnv(file, benchmark_dir, cell_column_name, norm):
    data = get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64')
    return data

def get_cell_mutations(file, benchmark_dir, cell_column_name, norm):
    data = get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64')
    return data

def get_drug_smiles(file, benchmark_dir, drug_column_name, norm=None):
    data = get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='str')
    return data

def get_drug_mordred(file, benchmark_dir, drug_column_name, norm=None):
    data = get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='float64')
    return data

def get_drug_infomax(file, benchmark_dir, drug_column_name, norm=None):
    data = get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='float64')
    return data

def get_drug_ecfp(file, benchmark_dir, drug_column_name, norm=None):
    data = get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='int')
    return data

def normalize_cell_features(df, norm_list):
    norm_df = df
    if (norm_list != []) and (norm_list != None):
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
                    norm_df = normalize_features(df, subtype)
                elif strategy == 'subset':
                    print(f"Running {strategy} with {subtype}.")
                    norm_df = subset_features(df, subtype)
    return norm_df

def normalize_features(df, subtype):
    norm_df = df
    if subtype == 'zscale':
        norm_df = z_scale_dataframe(df.copy())
    else:
        print("zscale is the only implemented normalization")
    return norm_df

def subset_features(df, subtype):
    norm_df = df
    if subtype == 'high_variance':
        norm_df = subset_high_variance(df)
    elif subtype == 'L1000':
        norm_df = subset_L1000(df)
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

# normalize features
# RULES: dfs must always have ID as index
def subset_L1000(df):
    # FUTURE: this could take a parameter for any subset
    inter_list = list(set(L1000_ENTREZ) & set(df.columns.to_list()))
    sub_df = df[inter_list]
    return sub_df

def subset_high_variance(df, var_threshold=0.8):
    vars = df.var()
    vars_subset = vars[vars < var_threshold]
    df_subset = df[vars_subset.index]
    return df_subset

# merge functions
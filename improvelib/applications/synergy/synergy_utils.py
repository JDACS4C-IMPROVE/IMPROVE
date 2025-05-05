"""
This module provides utilities for loading and processing response data for 
synergy models in the IMPROVE framework.
"""

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
from improvelib.utils_app_generic import _get_full_input_path, _get_stage_splits, _get_all_splits, _get_x_data



##### Y_DATA FUNCTIONS #####
def get_response_data(split_file, benchmark_dir, response_file='synergy.tsv', sep='\t'):
    """Gets response data for a given split file.

    Args:
        split_file (Union[str, Path, list of str, list of Path]): Name of split file if in benchmark data, otherwise path to split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        response_file (str): Name of response file (default: 'synergy.tsv')
        sep (str): Separator for response file (default: '\t').

    Returns:
        pd.DataFrame: Response dataframe for given split.
    """
    # get path to y_data file, read data
    response_path = _get_full_input_path(response_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    # ensures the first four columns (cell ID, drug ID, drug ID, study) are strings
    df[df.columns[0:4]] = df[df.columns[0:4]].astype(str)
    # ensures the rest of the columns are floats
    df[df.columns[4:]] = df[df.columns[4:]].astype(float)
    # get path to splits file, read data
    ids = _get_stage_splits(split_file, benchmark_dir)
    # subset y_data based on split given
    df = df.loc[ids]
    return df

def get_all_response_data(train_split_file, val_split_file, test_split_file, benchmark_dir, response_file='synergy.tsv', sep='\t'):
    """Gets response data for all given split file. Denotes stage of split in col 'split' with 'train', 'val', or 'test'.

    Args:
        train_split_file (Union[str, Path, list of str, list of Path]): Name of train split file if in benchmark data, otherwise path to train split file. Can be a list of str or Path.
        val_split_file (Union[str, Path, list of str, list of Path]): Name of val split file if in benchmark data, otherwise path to val split file. Can be a list of str or Path.
        test_split_file (Union[str, Path, list of str, list of Path]): Name of test split file if in benchmark data, otherwise path to test split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        response_file (str): Name of response file (default: 'synergy.tsv')
        sep (str): Separator for response file (default: '\t').

    Returns:
        pd.DataFrame: Response dataframe for all splits with col 'split' denoting split type ('train', 'val', or 'test').
    """
    # get path to y_data file, read data
    response_path = _get_full_input_path(response_file, benchmark_dir, 'y_data')
    df = pd.read_csv(response_path, sep=sep)
    # ensures the first four columns (cell ID, drug ID, drug ID, study) are strings
    df[df.columns[0:4]] = df[df.columns[0:4]].astype(str)
    # ensures the rest of the columns are floats
    df[df.columns[4:]] = df[df.columns[4:]].astype(float)
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



#### X_DATA FUNCTIONS


def get_cell_transcriptomics(file, benchmark_dir, cell_column_name, norm, gene_id='Entrez'):
    """Gets cell transcriptomics. Sets index to cell ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of cell transcriptomics file if in benchmark data, otherwise path to cell transcriptomics file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for cell data.
        norm (list): Normalization to perform on this data.

    Returns:
        pd.DataFrame: cell transcriptomics data (with normalization if specified), index set to cell ID.
    """
    data = _get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64', gene_id=gene_id)
    return data

def get_cell_cnv(file, benchmark_dir, cell_column_name, norm, gene_id='Entrez'):
    """Gets cell Copy Number Variation. Sets index to cell ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of cell Copy Number Variation file if in benchmark data, otherwise path to cell Copy Number Variation file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for cell data.
        norm (list): Normalization to perform on this data.

    Returns:
        pd.DataFrame: cell Copy Number Variation data (with normalization if specified), index set to cell ID.
    """
    data = _get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64', gene_id=gene_id)
    return data

def get_cell_mutations(file, benchmark_dir, cell_column_name, norm, gene_id='Entrez'):
    """Gets cell mutation. Sets index to cell ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of cell mutation file if in benchmark data, otherwise path to cell mutation file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for cell data.
        norm (list): Normalization to perform on this data.

    Returns:
        pd.DataFrame: cell mutation data (with normalization if specified), index set to cell ID.
    """
    data = _get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64', gene_id=gene_id)
    return data

def get_drug_smiles(file, benchmark_dir, drug_column_name, norm=None):
    """Gets drug SMILES. Sets index to drug ID and sets dtype to str.

    Args:
        file (Union[str, Path]): Name of drug SMILES file if in benchmark data, otherwise path to drug SMILES file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for drug data.
        norm (list): None. No normalization is currently supported for SMILES data.

    Returns:
        pd.DataFrame: drug SMILES data, index set to drug ID.
    """
    data = _get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='str')
    return data

def get_drug_mordred(file, benchmark_dir, drug_column_name, norm=None):
    """Gets drug Mordred. Sets index to drug ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of drug Mordred file if in benchmark data, otherwise path to drug Mordred file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for drug data.
        norm (list): None. No normalization is currently supported for Mordred data.

    Returns:
        pd.DataFrame: drug Mordred data, index set to drug ID.
    """
    data = _get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='float64')
    return data

def get_drug_infomax(file, benchmark_dir, drug_column_name, norm=None):
    """Gets drug Infomax. Sets index to drug ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of drug Infomax file if in benchmark data, otherwise path to drug Infomax file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for drug data.
        norm (list): None. No normalization is currently supported for Infomax data.

    Returns:
        pd.DataFrame: drug Infomax data, index set to drug ID.
    """
    data = _get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='float64')
    return data

def get_drug_ecfp(file, benchmark_dir, drug_column_name, norm=None):
    """Gets drug Extended Connectivity FingerPrints (ECFP). Sets index to drug ID and sets dtype to int.

    Args:
        file (Union[str, Path]): Name of drug ECFP file if in benchmark data, otherwise path to drug ECFP file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for drug data.
        norm (list): None. No normalization is currently supported for ECFP data.

    Returns:
        pd.DataFrame: drug ECFP data, index set to drug ID.
    """
    data = _get_x_data(file, benchmark_dir, drug_column_name, norm, dtype='int')
    return data





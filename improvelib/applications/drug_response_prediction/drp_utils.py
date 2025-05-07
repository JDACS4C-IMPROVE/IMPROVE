"""
This module provides utilities for loading and processing response data for 
drug response prediction models in the IMPROVE framework. It also provides 
functionality to filter dataframes, retaining only the common IDs shared between them.
"""

# Standard library imports
from ast import literal_eval
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from improvelib.utils_app_generic import _get_full_input_path, _get_stage_splits, _get_all_splits, _get_x_data
from improvelib.statics import L1000_ENTREZ, L1000_SYMBOL
from improvelib.applications.drug_response_prediction.drp_statics import methyl_symbol_dict, methyl_entrez_dict, methyl_ensembl_dict

# Set logger for this module
FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))


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


def common_elements(list1: List, list2: List, verbose: bool = False) -> List:
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

def get_response_data(split_file, benchmark_dir, response_file='response.tsv', sep='\t'):
    """Gets response data for a given split file.

    Args:
        split_file (Union[str, Path, list of str, list of Path]): Name of split file if in benchmark data, otherwise path to split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        response_file (str): Name of response file (default: 'response.tsv')
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

def get_all_response_data(train_split_file, val_split_file, test_split_file, benchmark_dir, response_file='response.tsv', sep='\t'):
    """Gets response data for all given split file. Denotes stage of split in col 'split' with 'train', 'val', or 'test'.

    Args:
        train_split_file (Union[str, Path, list of str, list of Path]): Name of train split file if in benchmark data, otherwise path to train split file. Can be a list of str or Path.
        val_split_file (Union[str, Path, list of str, list of Path]): Name of val split file if in benchmark data, otherwise path to val split file. Can be a list of str or Path.
        test_split_file (Union[str, Path, list of str, list of Path]): Name of test split file if in benchmark data, otherwise path to test split file. Can be a list of str or Path.
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        response_file (str): Name of response file (default: 'response.tsv')
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

def get_cell_transcriptomics(file, benchmark_dir, cell_column_name, norm, gene_id='Symbol'):
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

def get_cell_cnv(file, benchmark_dir, cell_column_name, norm, gene_id='Symbol'):
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

def get_cell_mutations(file, benchmark_dir, cell_column_name, norm, gene_id='Symbol'):
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

def get_cell_methylation(file, benchmark_dir, cell_column_name, norm):
    """Gets cell mutation. Sets index to cell ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of cell mutation file if in benchmark data, otherwise path to cell mutation file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for cell data.
        norm (list): Normalization to perform on this data.

    Returns:
        pd.DataFrame: cell mutation data (with normalization if specified), index set to cell ID.
    """
    data = _get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64')
    return data

def get_cell_miRNA(file, benchmark_dir, cell_column_name, norm):
    """Gets cell mutation. Sets index to cell ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of cell mutation file if in benchmark data, otherwise path to cell mutation file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for cell data.
        norm (list): Normalization to perform on this data.

    Returns:
        pd.DataFrame: cell mutation data (with normalization if specified), index set to cell ID.
    """
    data = _get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64')
    return data

def get_cell_RPPA(file, benchmark_dir, cell_column_name, norm):
    """Gets cell mutation. Sets index to cell ID and sets dtype to float64.

    Args:
        file (Union[str, Path]): Name of cell mutation file if in benchmark data, otherwise path to cell mutation file. 
        benchmark_dir (Union[str, Path]): Path to benchmark data directory.
        cell_column_name (str): Name of ID column for cell data.
        norm (list): Normalization to perform on this data.

    Returns:
        pd.DataFrame: cell mutation data (with normalization if specified), index set to cell ID.
    """
    data = _get_x_data(file, benchmark_dir, cell_column_name, norm, dtype='float64')
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

def change_gene_identifiers(data, data_type, identifier):
    if data_type == 'methyl' or data_type == 'methylation':
        if identifier == 'Symbol' or identifier == 'symbol' or identifier == 'gene_symbol':
            data = data.rename(columns=methyl_symbol_dict)
        elif identifier == 'Entrez' or identifier == 'entrez':
            data = data.rename(columns=methyl_entrez_dict)
        elif identifier == 'Ensembl' or identifier == 'ensembl':
            data = data.rename(columns=methyl_ensembl_dict)
        else:
            raise ValueError(f"ERROR! Identfied provided was {identifier} but must be one of 'Entrez' or 'Symbol' or 'Ensembl'.\n")
    else:
        print("Only methylation has been implemented.")
    return data




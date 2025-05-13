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

from improvelib.utils_app_generic import get_x_data, get_response_data, get_all_response_data, determine_transform, transform_data, get_response_with_features, get_features_in_response
from improvelib.statics import L1000_ENTREZ, L1000_SYMBOL, LINCS_SYMBOL
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




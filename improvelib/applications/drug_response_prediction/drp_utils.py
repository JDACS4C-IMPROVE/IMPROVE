"""
This module provides utilities for loading and processing response data for 
drug response prediction models in the IMPROVE framework. It also provides 
functionality to filter dataframes, retaining only the common IDs shared between them.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging
import os
from ast import literal_eval
import pandas as pd

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
        df1: First dataframe.
        df2: Second dataframe.
        ref_col: The reference column to find the common values.

    Returns:
        Tuple of DataFrames after filtering for common data.
    """
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def common_elements(list1: List, list2: List, verbose: bool = False) -> List:
    """Return a list of elements that the provided lists have in common.

    Args:
        list1: One list.
        list2: Another list.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default is False.

    Returns:
        List of common elements.
    """
    in_common = list(set(list1).intersection(set(list2)))
    if verbose:
        print("Elements in common count: ", len(in_common))
    return in_common


class DrugResponseLoader:
    """Class for loading monotherapy drug response data.

    Args:
        params: IMPROVE parameters.
        split_file: File name that contains the split ids (rows).
        sep: Character separator in the loaded files (e.g., "\t" for TSV files).
        verbose: Flag for verbosity. Default is True.

    Attributes:
        response_fname: Default response file name.
        known_file_names: List of known file names.
        params: Parameters for loading data.
        sep: Separator used in data files.
        inp: Parsed input data files.
        y_col_name: Column name for the target variable.
        canc_col_name: Column name for cancer sample identifiers.
        drug_col_name: Column name for drug identifiers.
        y_data_path: Path to the directory containing y data files.
        split_fpath: Path to the file containing split identifiers.
        dfs: Dictionary to store loaded dataframes.
        verbose: Verbosity flag.

    Example:
        from improve import drug_resp_pred as drp
        drp_loader = drp.DrugResponseLoader(params)
        print(drp_loader)
        print(dir(drp_loader))
        rsp = drp_loader["response.tsv"]
    """

    def __init__(self, params: Dict, split_file: str, sep: str = "\t", verbose: bool = True):
        self.response_fname = "response.tsv"  # Default response file name
        self.known_file_names = [self.response_fname]  # List of known file names

        self.params = params  # Parameters for loading data
        self.sep = sep  # Separator used in data files
        if isinstance(params["y_data_files"], str):
            self.inp = literal_eval(params["y_data_files"])  # Parsed input data files
        else:
            self.inp = params["y_data_files"]

        self.y_col_name = params["y_col_name"]  # Column name for the target variable
        self.canc_col_name = params["canc_col_name"]  # Column name for cancer sample identifiers
        self.drug_col_name = params["drug_col_name"]  # Column name for drug identifiers

        self.y_data_path = params["y_data_path"]  # Path to the directory containing y data files
        self.split_fpath = Path(params["splits_path"]) / split_file  # Path to the file containing split identifiers
        self.dfs = {}  # Dictionary to store loaded dataframes
        self.verbose = verbose  # Verbosity flag

        if self.verbose:
            print(f"y_data_files: {params['y_data_files']}")
            print(f"y_col_name: {params['y_col_name']}")

        self.load_all_response_data()

    def __repr__(self) -> str:
        """Return a string representation of the loaded data."""
        if self.dfs:
            to_print = []
            to_print.append("Loaded data:\n")
            to_print.append("\n".join(
                [f"{fname}: {df.shape} \nUnique cells: {df[self.canc_col_name].nunique()} \nUnique drugs: {df[self.drug_col_name].nunique()}" for fname, df in self.dfs.items()]))
            to_print = "".join(to_print)
            return to_print
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath: str):
        """Check if the given file path exists.

        Args:
            fpath: File path to check.

        Raises:
            Exception: If the file path does not exist.
        """
        fpath = Path(fpath)
        if not fpath.exists():
            raise Exception(f"ERROR! {fpath} not found.\n")

    def load_response_data(self, fname: str) -> pd.DataFrame:
        """Load response data from a file.

        Args:
            fname: File name to load data from.

        Returns:
            Loaded data as a DataFrame.
        """
        fpath = Path(os.path.join(str(self.y_data_path), fname))
        logger.debug(f"Loading {fpath}")
        DrugResponseLoader.check_path(fpath)
        df = pd.read_csv(fpath, sep=self.sep)
        return df

    def load_all_response_data(self):
        """Load all response data files specified in the input parameters."""
        for i in self.inp[0]:
            fname = i
            df = self.load_response_data(fname)
            DrugResponseLoader.check_path(self.split_fpath)
            ids = pd.read_csv(self.split_fpath, header=None)[0].tolist()
            df = df.loc[ids]
            self.dfs[fname] = df


"""
This module provides utilities for loading and processing drug data
for drug response prediction models in the IMPROVE framework.
"""

from ast import literal_eval
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


# Set logger for this module
FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))


class DrugsLoader:
    """Class to load and manage drug data.

    Args:
        params (Dict): IMPROVE parameters.
        sep (str): Character separator in the loaded files (e.g., "\t" for TSV files).
        verbose (bool): Whether to print detailed loading information.

    Attributes:
        smiles_fname (str): Filename for SMILES data.
        mordred_fname (str): Filename for Mordred descriptors.
        ecfp4_512bit_fname (str): Filename for ECFP4 512-bit data.
        known_file_names (List[str]): List of known drug data filenames.
        params (Dict): Configuration parameters for loading data.
        sep (str): Separator used in data files.
        inp (List): List of input drug data files.
        drug_col_name (str): Column name for drug identifiers.
        x_data_path (str): Path to the directory containing drug data files.
        dfs (Dict[str, pd.DataFrame]): Dictionary to store loaded dataframes, keyed by filename.
        verbose (bool): Flag to control verbosity of output.
        inp_fnames (List[str]): List of input filenames extracted from inp.

    Example:
        from improve import drug_resp_pred as drp
        params = {
            "x_data_drug_files": "[['drug_SMILES.tsv'], ['drug_mordred.tsv'], ['drug_ecfp4_nbits512.tsv']]",
            "drug_col_name": "DrugID",
            "x_data_path": "/path/to/drug/data"
        }
        drugs_loader = drp.DrugsLoader(params)
        print(drugs_loader)
        print(dir(drugs_loader))
        smi = drugs_loader["drug_SMILES.tsv"]
    """

    def __init__(self, params: Dict, sep: str = "\t", verbose: bool = True):
        """Initialize the DrugsLoader with parameters and file settings.

        Args:
            params (Dict): IMPROVE parameters.
            sep (str): Character separator in the loaded files (e.g., "\t" for TSV files).
            verbose (bool): Whether to print detailed loading information.
        """
        
        # Filenames for different types of drug data
        self.smiles_fname = "drug_SMILES.tsv"
        self.mordred_fname = "drug_mordred.tsv"
        self.ecfp4_512bit_fname = "drug_ecfp4_nbits512.tsv"

        # List of known drug data filenames
        self.known_file_names = [
            self.smiles_fname,
            self.mordred_fname,
            self.ecfp4_512bit_fname
        ]

        self.params = params
        self.sep = sep

        # Convert input files from string to list if necessary
        if isinstance(params["x_data_drug_files"], str):
            self.inp = literal_eval(params["x_data_drug_files"])
        else:
            self.inp = params["x_data_drug_files"]

        logger.debug(f"self.inp: {self.inp}")

        self.drug_col_name = params["drug_col_name"]
        self.x_data_path = params["x_data_path"]
        self.dfs = {}
        self.verbose = verbose

        if self.verbose:
            print(f"drug_col_name: {params['drug_col_name']}")
            print("x_data_drug_files:")
            for i, d in enumerate(self.inp):
                print(f"{i+1}. {d}")

        # Extract filenames from input list
        self.inp_fnames = []
        for i in self.inp:
            assert len(i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])

        self.load_all_drug_data()

    def __repr__(self) -> str:
        """Return a string representation of the loaded data."""
        if self.dfs:
            return "Loaded data:\n" + "\n".join(
                [f"{fname}: {df.shape}" for fname, df in self.dfs.items()]
            )
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath: Union[str, Path]) -> None:
        """Check if a file path exists.

        Args:
            fpath (Union[str, Path]): Path to check.

        Raises:
            Exception: If the path does not exist.
        """
        fpath = Path(fpath)
        if not fpath.exists():
            raise Exception(f"ERROR! {fpath} not found.\n")

    def load_drug_data(self, fname: str) -> pd.DataFrame:
        """Load a single drug data file.

        Args:
            fname (str): Filename of the drug data file to load.

        Returns:
            pd.DataFrame: Loaded drug data as a DataFrame.
        """
        fpath = os.path.join(self.x_data_path, fname)
        DrugsLoader.check_path(fpath)
        df = pd.read_csv(fpath, sep=self.sep)
        df = df.set_index(self.drug_col_name)
        return df

    def load_all_drug_data(self) -> None:
        """Load all drug data specified in self.inp."""
        for i in self.inp:
            fname = i[0]
            self.dfs[fname] = self.load_drug_data(fname)

        print("Finished loading drug data.")

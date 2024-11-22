"""
This module provides utilities for loading and processing omics data
for drug response prediction models in the IMPROVE framework.

Omics data files are generally multi-level tables with several column types
(typically 3 or 4), containing identifiers relevant to the data type, such 
as gene-related identifiers (Entrez ID, Gene Symbol, Ensembl ID, TSS) or 
other domain-specific identifiers for proteins, metabolites, etc.
The `level_map` dictionary in each loader function encodes the
column level and the corresponding identifier systems.

Files that use `level_map`:
    - cancer_copy_number.tsv
    - cancer_discretized_copy_number.tsv
    - cancer_DNA_methylation.tsv
    - cancer_gene_expression.tsv
    - cancer_miRNA_expression.tsv
    - cancer_mutation_count.tsv
    - cancer_RPPA.tsv

Exceptions (files that do not use `level_map` and may not be multi-level):
    - cancer_mutation_long_format.tsv
    - cancer_mutation.parquet
    
Notes:
    - Ensure that the input omics data files are correctly formatted as 
      multi-level tables, except for the noted exceptions.
    - The `level_map` must be correctly defined for each type of omics data 
      file to ensure proper column renaming.
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


def set_col_names_in_multilevel_dataframe(
        df: pd.DataFrame,
        level_map: Dict,
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol") -> pd.DataFrame:
    """Support loading of omic data files by renaming multi-level column names.

    Args:
        df (pd.DataFrame): Omics dataframe.
        level_map (Dict): Encodes the column level and the corresponding 
            identifier systems.
        gene_system_identifier (Union[str, List[str]]): Gene identifier system 
            to use. Options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any 
            list combination of ["Entrez", "Gene_Symbol", "Ensembl"].

    Returns:
        pd.DataFrame: The input dataframe with the specified multi-level column names.
    """
    df = df.copy()

    level_names = list(level_map.keys())
    level_values = list(level_map.values())
    n_levels = len(level_names)

    if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
        gene_system_identifier = gene_system_identifier[0]

    if isinstance(gene_system_identifier, str):
        if gene_system_identifier == "all":
            # Assign multi-level column names
            df.columns = df.columns.rename(level_names, level=level_values)
        else:
            df.columns = df.columns.get_level_values(
                # Retain specific column level
                level_map[gene_system_identifier])
    else:
        if len(gene_system_identifier) > n_levels:
            raise Exception(
                f"ERROR! 'gene_system_identifier' can't contain more than {n_levels} items.\n")
        set_diff = list(
            set(gene_system_identifier).difference(set(level_names)))
        if len(set_diff) > 0:
            raise Exception(
                f"ERROR! Passed unknown gene identifiers: {set_diff}.\n")
        kk = {i: level_map[i]
              for i in level_map if i in gene_system_identifier}
        # Assign multi-level column names
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())
        drop_levels = list(
            set(level_map.values()).difference(set(kk.values())))
        df = df.droplevel(level=drop_levels, axis=1)
    return df


class OmicsLoader:
    """Class aggregates methods to load omics data.

    Args:
        params (Dict): IMPROVE parameters.
        sep (str): Character separator in the loaded files (e.g., "\t" for TSV files).
        verbose (bool): Flag for verbosity.

    Attributes:
        copy_number_fname (str): Filename for copy number data.
        discretized_copy_number_fname (str): Filename for discretized copy number data.
        dna_methylation_fname (str): Filename for DNA methylation data.
        gene_expression_fname (str): Filename for gene expression data.
        miRNA_expression_fname (str): Filename for miRNA expression data.
        mutation_count_fname (str): Filename for mutation count data.
        mutation_long_format_fname (str): Filename for mutation data in long format.
        mutation_fname (str): Filename for mutation data in parquet format.
        rppa_fname (str): Filename for RPPA data.
        known_file_names (List[str]): List of known omics data filenames.
        params (Dict): Configuration parameters for loading data.
        sep (str): Separator used in data files.
        inp (List): List of input omics data files.
        x_data_path (str): Path to the directory containing omics data files.
        canc_col_name (str): Column name for indexing in the data files.
        dfs (Dict[str, pd.DataFrame]): Dictionary to store loaded dataframes, keyed by filename.
        verbose (bool): Flag to control verbosity of output.
        inp_fnames (List[str]): List of input filenames extracted from inp.

    Example:
        from improve import drug_resp_pred as drp
        params = {
            "x_data_canc_files": "[['cancer_gene_expression.tsv', 'Gene_Symbol'], "
                                 "['cancer_copy_number.tsv', 'Entrez']]",
            "canc_col_name": "SampleID",
            "x_data_path": "/path/to/omics/data"
        }
        omics_loader = drp.OmicsLoader(params)
        print(omics_loader)
        print(dir(omics_loader))
        gene_expression_data = omics_loader["cancer_gene_expression.tsv"]
    """

    def __init__(self, params: Dict, sep: str = "\t", verbose: bool = True):
        """Initialize the OmicsLoader with parameters and file settings."""
        
        # Filenames for different types of omics data
        self.copy_number_fname = "cancer_copy_number.tsv"
        self.discretized_copy_number_fname = "cancer_discretized_copy_number.tsv"
        self.dna_methylation_fname = "cancer_DNA_methylation.tsv"
        self.gene_expression_fname = "cancer_gene_expression.tsv"
        self.miRNA_expression_fname = "cancer_miRNA_expression.tsv"
        self.mutation_count_fname = "cancer_mutation_count.tsv"
        self.mutation_long_format_fname = "cancer_mutation_long_format.tsv"
        self.mutation_fname = "cancer_mutation.parquet"
        self.rppa_fname = "cancer_RPPA.tsv"

        # List of known omics data filenames
        self.known_file_names = [
            self.copy_number_fname,
            self.discretized_copy_number_fname,
            self.dna_methylation_fname,
            self.gene_expression_fname,
            self.miRNA_expression_fname,
            self.mutation_count_fname,
            self.mutation_long_format_fname,
            self.mutation_fname,
            self.rppa_fname
        ]

        self.params = params
        self.sep = sep
        
        # Convert input files from string to list if necessary
        if isinstance(params["x_data_canc_files"], str):
            logger.debug("x_data_canc_files is a string. Converting to list.")
            self.inp = literal_eval(params["x_data_canc_files"])
        else:
            self.inp = params["x_data_canc_files"]

        logger.debug(f"self.inp: {self.inp}")

        self.x_data_path = params["x_data_path"]
        self.canc_col_name = params["canc_col_name"]
        self.dfs = {}
        self.verbose = verbose

        if self.verbose:
            print(f"canc_col_name: {params['canc_col_name']}")
            print("x_data_canc_files:")
            for i, o in enumerate(self.inp):
                print(f"{i+1}. {o}")

        # Extract filenames from input list
        self.inp_fnames = []
        for i in self.inp:
            assert 0 < len(i) < 3, f"Inner lists must contain one or two items, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])

        self.load_all_omics_data()

    def __repr__(self) -> str:
        """Return a string representation of the loaded data."""
        if self.dfs:
            return "Loaded data:\n" + "\n".join([f"{fname}: {df.shape}" for fname, df in self.dfs.items()])
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

    def load_all_omics_data(self) -> None:
        """Load all omics data specified in self.inp."""
        logger.info("Loading omics data.")
        for i in self.inp:
            fname = i[0]
            gene_system_identifier = i[1] if len(i) > 1 else "all"

            if fname == self.copy_number_fname:
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.discretized_copy_number_fname:
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.dna_methylation_fname:
                level_map = {"Ensembl": 2, "Entrez": 1, "Gene_Symbol": 3, "TSS": 0}
            elif fname == self.gene_expression_fname:
                level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
            elif fname == self.miRNA_expression_fname:
                level_map = {"miRNA_ID": 0}
            elif fname == self.mutation_count_fname:
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.mutation_long_format_fname:
                level_map = None  # level_map is not used with long format omics files
            elif fname == self.mutation_fname:
                level_map = None  # level_map is not used with parquet file
            elif fname == self.rppa_fname:
                level_map = {"Antibody": 0}
            else:
                raise NotImplementedError(f"Option '{fname}' not recognized.")

            fpath = os.path.join(self.x_data_path, fname)
            OmicsLoader.check_path(fpath)

            if self.verbose:
                print(f"Loading {fpath}")

            if fname.endswith(".parquet"):
                df = pd.read_parquet(fpath)
            elif "long_format" in fname:
                df = pd.read_csv(fpath, sep=self.sep)
            else:
                header = [i for i in range(len(level_map))]
                df = pd.read_csv(fpath, sep=self.sep, index_col=0, header=header)

            if level_map is not None:
                df.index.name = self.canc_col_name
                df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
                df = df.reset_index()

            self.dfs[fname] = df

        logger.info("Finished loading omics data.")


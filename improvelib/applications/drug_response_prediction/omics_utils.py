""" Functionality for IMPROVE drug response prediction (DRP) models. """

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

from ast import literal_eval

import pandas as pd


# Set logger for this module

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))


# ==============================================================
# Omic feature loaders
# ==============================================================

"""
Omics data files are multi-level tables with several column types (generally 3
or 4), each contains gene names using a different gene identifier system:
Entrez ID, Gene Symbol, Ensembl ID, TSS

The column levels are not organized in the same order across the different
omic files.

The level_map dict, in each loader function, encodes the column level and the
corresponding identifier systems.

For example, in the copy number file the level_map is:
level_map = {"Entrez":0, "Gene_Symbol": 1, "Ensembl": 2}
"""


def set_col_names_in_multilevel_dataframe(
        df: pd.DataFrame,
        level_map: Dict,
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol") -> pd.DataFrame:
    """ Util function that supports loading of the omic data files.
    Returns the input dataframe with the multi-level column names renamed as
    specified by the gene_system_identifier arg.

    Args:
        df (pd.DataFrame): omics dataframe
        level_map (dict): encodes the column level and the corresponding identifier systems
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: the input dataframe with the specified multi-level column names
    """
    df = df.copy()

    level_names = list(level_map.keys())
    level_values = list(level_map.values())
    n_levels = len(level_names)

    if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
        gene_system_identifier = gene_system_identifier[0]

    if isinstance(gene_system_identifier, str):
        if gene_system_identifier == "all":
            # assign multi-level col names
            df.columns = df.columns.rename(level_names, level=level_values)
        else:
            df.columns = df.columns.get_level_values(
                # retain specific column level
                level_map[gene_system_identifier])
    else:
        if len(gene_system_identifier) > n_levels:
            raise Exception(
                f"ERROR ! 'gene_system_identifier' can't contain more than {n_levels} items.\n")
        set_diff = list(
            set(gene_system_identifier).difference(set(level_names)))
        if len(set_diff) > 0:
            raise Exception(
                f"ERROR ! Passed unknown gene identifiers: {set_diff}.\n")
        kk = {i: level_map[i]
              for i in level_map if i in gene_system_identifier}
        # assign multi-level col names
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())
        drop_levels = list(
            set(level_map.values()).difference(set(kk.values())))
        df = df.droplevel(level=drop_levels, axis=1)
    return df


class OmicsLoader():
    """
    Class aggregates methods to load omics data.

    Args:
        params : IMPROVE params
        sep : chararcter separtor in the loaded files (e.g., "\t" for tsv files)

    Example:
        from improve import drug_resp_pred as drp
        omics_loader = drp.OmicsLoader(params)
        print(omics_loader)
        print(dir(omics_loader))
        ge = omics_loader["cancer_gene_expression.tsv"]
    """

    def __init__(self,
                 params: Dict,  # improve params
                 sep: str = "\t",
                 verbose: bool = True):

        self.copy_number_fname = "cancer_copy_number.tsv"
        self.discretized_copy_number_fname = "cancer_discretized_copy_number.tsv"
        self.dna_methylation_fname = "cancer_DNA_methylation.tsv"
        self.gene_expression_fname = "cancer_gene_expression.tsv"
        self.miRNA_expression_fname = "cancer_miRNA_expression.tsv"
        self.mutation_count_fname = "cancer_mutation_count.tsv"
        self.mutation_long_format_fname = "cancer_mutation_long_format.tsv"
        self.mutation_fname = "cancer_mutation.parquet"
        self.rppa_fname = "cancer_RPPA.tsv"
        self.known_file_names = [self.copy_number_fname,
                                 self.discretized_copy_number_fname,
                                 self.dna_methylation_fname,
                                 self.gene_expression_fname,
                                 self.miRNA_expression_fname,
                                 self.mutation_count_fname,
                                 self.mutation_long_format_fname,
                                 self.mutation_fname,
                                 self.rppa_fname]

        self.params = params
        self.sep = sep


        if isinstance(params["x_data_canc_files"], str):
            # instanciate array from string
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
            print(f"x_data_canc_files:")
            for i, o in enumerate(self.inp):
                print(f"{i+1}. {o}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(i) > 0 and len(
                i) < 3, f"Inner lists must contain one or two items, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])
        # print(self.inp_fnames)

        self.load_all_omics_data()

    def __repr__(self):
        if self.dfs:
            return "Loaded data:\n" + "\n".join([f"{fname}: {df.shape}" for fname, df in self.dfs.items()])
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath):
        fpath = Path(fpath)
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_all_omics_data(self):
        """ Load all omics data appear in self.inp """
        logger.info("Loading omics data.")
        for i in self.inp:
            fname = i[0]
            if len(i) > 1:
                gene_system_identifier = i[1]
            else:
                gene_system_identifier = "all"

            if fname == self.copy_number_fname:
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.discretized_copy_number_fname:
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.dna_methylation_fname:
                level_map = {"Ensembl": 2, "Entrez": 1,
                             "Gene_Symbol": 3, "TSS": 0}
            elif fname == self.gene_expression_fname:
                level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
            elif fname == self.miRNA_expression_fname:
                level_map = {"miRNA_ID": 0}
            elif fname == self.mutation_count_fname:
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.mutation_long_format_fname:
                level_map = None  # no levels in long format omics files
            elif fname == self.mutation_fname:
                level_map = None  # level_map is not used with parquet file
            elif fname == self.rppa_fname:
                level_map = {"Antibody": 0}
            else:
                raise NotImplementedError(f"Option '{fname}' not recognized.")

            # fpath = imp_glob.X_DATA_DIR / fname
            fpath = os.path.join(self.x_data_path, fname)
            OmicsLoader.check_path(fpath)

            if self.verbose:
                print(f"Loading {fpath}")

            if fname.split(".")[-1] == "parquet":
                df = pd.read_parquet(fpath)
            elif "long_format" in fname:
                df = pd.read_csv(fpath, sep=self.sep)
            else:
                # header is used for multilevel omics files
                header = [i for i in range(len(level_map))]
                df = pd.read_csv(fpath, sep=self.sep,
                                 index_col=0, header=header)

            if level_map is not None:
                # if "long_format" not in fname:
                # print(df.iloc[:4, :5])
                df.index.name = self.canc_col_name  # assign index name
                # print(df.iloc[:4, :5])
                df = set_col_names_in_multilevel_dataframe(
                    df, level_map, gene_system_identifier)
                # print(df.iloc[:4, :4])
                df = df.reset_index()
                # print(df.iloc[:4, :5])

            self.dfs[fname] = df

        # breakpoint()
        # print(self.dfs[self.copy_number_fname].iloc[:4, :4])
        # print(self.dfs[self.gene_expression_fname].iloc[:4, :4])
        # print(self.dfs[self.dna_methylation_fname].iloc[:4, :4])
        # print("Finished loading omics data.")
        logger.info("Finished loading omics data.")

""" Functionality for IMPROVE drug response prediction (DRP) models. """

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# from improve import framework as frm
from . import framework as frm
# from .framework import imp_glob  ## ap; removed

# ---------------------------------------------------------------
## File names that should be accesible to all IMPROVE DRP models.
# ---------------------------------------------------------------
# # Cancer sample features file names
# copy_number_fname = "cancer_copy_number.tsv"
# discretized_copy_number_fname = "cancer_discretized_copy_number.tsv"
# dna_methylation_fname = "cancer_DNA_methylation.tsv"
# gene_expression_fname = "cancer_gene_expression.tsv"
# miRNA_expression_fname = "cancer_miRNA_expression.tsv"
# mutation_count_fname = "cancer_mutation_count.tsv"
# mutation_fname = "cancer_mutation.tsv"
# rppa_fname = "cancer_RPPA.tsv"
# # Drug features file names
# smiles_fname = "drug_SMILES.tsv"
# mordred_fname = "drug_mordred.tsv"
# ecfp4_512bit_fname = "drug_ecfp4_512bit.tsv"


def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Search for common data in reference column and retain only .

    Args:
        df1, df2 (pd.DataFrame): dataframes
        ref_col (str): the ref column to find the common values

    Returns:
        df1, df2 after filtering for common data.

    Example:
        Before:

        df1:
        col1	ref_col	    col3	    col4
        CCLE	ACH-000956	Drug_749	0.7153
        CCLE	ACH-000325	Drug_1326	0.9579
        CCLE	ACH-000475	Drug_490	0.213

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327	0.0976107966264223	4.888499735514123
        ACH-000179  5.202025844609336	3.5046203924035524	3.5058909297299574
        ACH-000325  6.016139702655253	0.6040713236688608	0.0285691521967709

        After:

        df1:
        col1	ref_col	    col3	    col4
        CCLE	ACH-000956	Drug_749	0.7153
        CCLE	ACH-000325	Drug_1326	0.9579

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327	0.0976107966264223	4.888499735514123
        ACH-000325  6.016139702655253	0.6040713236688608	0.0285691521967709
    """
    # Retain df1 and df2 samples with common ref_col
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def common_elements(list1: List, list2: List, verbose: bool=False) -> List:
    """
    Return list of elements that the provided lists have in common.

    Args:
        list1: One list.
        list2: Another list.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        List of common elements.
    """
    in_common = list(set(list1).intersection(set(list2)))
    if verbose:
        print("Elements in common count: ", len(in_common))
    return in_common


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
            df.columns = df.columns.rename(level_names, level=level_values)  # assign multi-level col names
        else:
            df.columns = df.columns.get_level_values(level_map[gene_system_identifier])  # retain specific column level
    else:
        if len(gene_system_identifier) > n_levels:
            raise Exception(f"ERROR ! 'gene_system_identifier' can't contain more than {n_levels} items.\n")
        set_diff = list(set(gene_system_identifier).difference(set(level_names)))
        if len(set_diff) > 0:
            raise Exception(f"ERROR ! Passed unknown gene identifiers: {set_diff}.\n")
        kk = {i: level_map[i] for i in level_map if i in gene_system_identifier}
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())  # assign multi-level col names
        drop_levels = list(set(level_map.values()).difference(set(kk.values())))
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
        self.mutation_fname = "cancer_mutation.tsv"
        self.rppa_fname = "cancer_RPPA.tsv"
        self.known_file_names = [self.copy_number_fname,
                                 self.discretized_copy_number_fname,
                                 self.dna_methylation_fname,
                                 self.gene_expression_fname,
                                 self.miRNA_expression_fname,
                                 self.mutation_count_fname,
                                 self.mutation_fname,
                                 self.rppa_fname]

        self.sep = sep
        self.inp = params["x_data_canc_files"]
        self.x_data_path = params["x_data_path"]
        self.canc_col_name = params["canc_col_name"]
        self.dfs = {}

        if verbose:
            print(f"x_data_canc_files: {params['x_data_canc_files']}")
            print(f"canc_col_name: {params['canc_col_name']}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(i) > 0 and len(i) < 3, f"Inner lists must contain one or two items, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])
        # print(self.inp_fnames)

        self.load_all_omics_data()

    def __repr__(self):
        if self.dfs:
            return "Loaded data\n" + "\n".join([f"{fname}: {df.shape}" for fname, df in self.dfs.items()])
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath):
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_all_omics_data(self):
        """ Load all omics data appear in self.inp """
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
                level_map = {"Ensembl": 2, "Entrez": 1, "Gene_Symbol": 3, "TSS": 0}
            elif fname == self.gene_expression_fname: 
                level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
            elif fname == self.miRNA_expression_fname: 
                raise NotImplementedError(f"{fname} not implemeted yet.")
            elif fname == self.mutation_count_fname: 
                level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
            elif fname == self.mutation_fname: 
                raise NotImplementedError(f"{fname} not implemeted yet.")
            elif fname == self.rppa_fname: 
                raise NotImplementedError(f"{fname} not implemeted yet.")
            else:
                raise NotImplementedError(f"Option '{fname}' not recognized.")

            # fpath = imp_glob.X_DATA_DIR / fname
            fpath = self.x_data_path / fname
            OmicsLoader.check_path(fpath)

            header = [i for i in range(len(level_map))]
            df = pd.read_csv(fpath, sep=self.sep, index_col=0, header=header)
            df.index.name = self.canc_col_name  # assign index name
            df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
            df = df.reset_index()
            self.dfs[fname] = df

        # import ipdb; ipdb.set_trace()
        # print(self.dfs[self.copy_number_fname].iloc[:4, :4])
        # print(self.dfs[self.gene_expression_fname].iloc[:4, :4])
        # print(self.dfs[self.dna_methylation_fname].iloc[:4, :4])
        # print("done")




# ==============================================================
# Drug feature loaders
# ==============================================================

class DrugsLoader():
    """
    Example:
        from improve import drug_resp_pred as drp
        drugs_loader = drp.DrugsLoader(params)
        print(drugs_loader)
        print(dir(drugs_loader))
        smi = drugs_loader["drug_SMILES.tsv"]
    """
    def __init__(self,
                 params: Dict,  # improve params
                 sep: str = "\t",
                 verbose: bool = True):

        self.smiles_fname = "drug_SMILES.tsv"
        self.mordred_fname = "drug_mordred.tsv"
        self.ecfp4_512bit_fname = "drug_ecfp4_nbits512.tsv"
        self.known_file_names = [self.smiles_fname,
                                 self.mordred_fname,
                                 self.ecfp4_512bit_fname]

        self.sep = sep
        self.inp = params["x_data_drug_files"]
        self.drug_col_name = params["drug_col_name"]
        self.x_data_path = params["x_data_path"]
        self.dfs = {}

        if verbose:
            print(f"x_data_drug_files: {params['x_data_drug_files']}")
            print(f"drug_col_name: {params['drug_col_name']}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])
        # print(self.inp_fnames)

        self.load_all_drug_data()

    def __repr__(self):
        if self.dfs:
            return "Loaded data\n" + "\n".join([f"{fname}: {df.shape}" for fname, df in self.dfs.items()])
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath):
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_drug_data(self, fname):
        """ ... """
        fpath = self.x_data_path / fname
        DrugsLoader.check_path(fpath)
        df = pd.read_csv(fpath, sep=self.sep)
        df = df.set_index(self.drug_col_name)
        return df

    def load_all_drug_data(self):
        """ ... """
        for i in self.inp:
            fname = i[0]
            self.dfs[fname] = self.load_drug_data(fname)

        # import ipdb; ipdb.set_trace()
        # print(self.dfs[self.smiles_fname].iloc[:4, :])
        # print(self.dfs[self.mordred_fname].iloc[:4, :4])
        # print(self.dfs[self.ecfp4_512bit_fname].iloc[:4, :4])
        # print("done")




# ==============================================================
# Drug response loaders
# ==============================================================


# source: str,
# # split: Union[int, None] = None,
# # split_type: Union[str, List[str], None] = None,
# split_file_name: Union[str, List[str], None] = None,
# y_col_name: str = "auc",
# sep: str = "\t",
# verbose: bool = True) -> pd.DataFrame:


class DrugResponseLoader():
    """
    Class for loading monotherapy drug response data.

    Args:
        params : IMPROVE params
        split_file : file name that contains the split ids (rows)
        sep : chararcter separtor in the loaded files (e.g., "\t" for tsv files)

    Example:
        from improve import drug_resp_pred as drp
        drp_loader = drp.DrugResponseLoader(params)
        print(drp_loader)
        print(dir(drp_loader))
        rsp = drp_loader["response.tsv"]
    """
    def __init__(self,
                 params: Dict,  # improve params
                 split_file: str,
                 sep: str = "\t",
                 verbose: bool = True):

        self.response_fname = "response.tsv"
        self.known_file_names = [self.response_fname]

        self.sep = sep
        self.inp = params["y_data_files"]
        self.y_col_name = params["y_col_name"]
        self.canc_col_name = params["canc_col_name"]
        self.drug_col_name = params["drug_col_name"]

        # self.y_data_path = params["y_data_path"]/params["y_data_files"][0][0]
        self.y_data_path = params["y_data_path"]
        self.split_fpath = params["splits_path"]/split_file
        self.dfs = {}

        if verbose:
            print(f"y_data_files: {params['y_data_files']}")
            print(f"y_col_name: {params['y_col_name']}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])
        # print(self.inp_fnames)

        self.load_all_response_data()

    def __repr__(self):
        if self.dfs:
            to_print = []
            to_print.append("Loaded data\n")
            to_print.append( "\n".join([f"{fname}: {df.shape} \nUnique cells: {df[self.canc_col_name].nunique()} \nUnique drugs: {df[self.drug_col_name].nunique()}" for fname, df in self.dfs.items()]) )
            to_print = "".join(to_print)
            return to_print
        else:
            return "No data files were loaded."

    @staticmethod
    def check_path(fpath):
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_response_data(self, fname):
        fpath = self.y_data_path / fname
        DrugResponseLoader.check_path(fpath)
        df = pd.read_csv(fpath, sep=self.sep)
        return df

    def load_all_response_data(self, verbose: str = True):
        """ ... """
        for i in self.inp:
            fname = i[0]
            df = self.load_response_data(fname)
            DrugResponseLoader.check_path(self.split_fpath)
            ids = pd.read_csv(self.split_fpath, header=None)[0].tolist()
            df = df.loc[ids]
            self.dfs[fname] = df

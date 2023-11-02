""" Functionality for IMPROVE drug response prediction (DRP) models. """

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# from improve import framework as frm
from . import framework as frm
# from .framework import imp_glob  ## ap; removed

# TODO: omic data files have different mappings for multi-level index mapping.
# level_map_cell_data = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}

## File names of should be accesible to all IMPROVE DRP models.
# Cancer sample features file names
copy_number_fname = "cancer_copy_number.tsv"  # cancer feature
discretized_copy_number_fname = "cancer_discretized_copy_number.tsv"  # cancer feature
dna_methylation_fname = "cancer_DNA_methylation.tsv"  # cancer feature
gene_expression_fname = "cancer_gene_expression.tsv"  # cancer feature
miRNA_expression_fname = "cancer_miRNA_expression.tsv"  # cancer feature
mutation_count_fname = "cancer_mutation_count.tsv"  # cancer feature
mutation_fname = "cancer_mutation.tsv"  # cancer feature
rppa_fname = "cancer_RPPA.tsv"  # cancer feature

# Drug features file names
smiles_fname = "drug_SMILES.tsv"  # drug feature
mordred_fname = "drug_mordred.tsv"  # drug feature
ecfp4_512bit_fname = "drug_ecfp4_512bit.tsv"  # drug feature


def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Search for common data in reference column and retain only .

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


# TODO(done). Renamed func to load_cell_data()
# def load_omics_data(fname: Union[Path, str],
# def load_omics_data(fname: Union[str, Path, List[str], List[Path]] = ["cancer_gene_expression.tsv"],

class OmicsLoader():
    """
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
        """ ... """
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
            print(f"canc_cole_name: {params['canc_col_name']}")

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
        """ ... """
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


def load_omics_data(params: Dict,
                    omics_type: str = "gene_expression",
                    canc_col_name: str = "improve_sample_id",
                    gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
                    use_lincs: bool = False,
                    sep: str = "\t",
                    verbose: bool = True) -> pd.DataFrame:
    """
    Returns dataframe with specified omics data.

    Args:
        fname: Name of, or Path to, file for reading cell data.
        canc_col_name: Column name that contains the cancer sample ids. Default: "improve_sample_id".
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        pd.DataFrame: dataframe with the cell line data.
    """

    # omics_file_to_name_mapping = {
    #     "cancer_copy_number.tsv": ["copy_number", {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}],
    #     "cancer_discretized_copy_number.tsv": ["discretized_copy_number", {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}],
    #     "cancer_DNA_methylation.tsv": ["dna_methylation", {"Ensembl": 2, "Entrez": 1, "Gene_Symbol": 3, "TSS": 0}],
    #     "cancer_gene_expression.tsv": ["gene_expression", {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}],
    #     "cancer_miRNA_expression.tsv": ["miRNA_expression", {}],
    #     "cancer_mutation_count.tsv": ["mutation_count", {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}],
    #     "cancer_mutation.tsv": ["mutation", {}],
    #     "cancer_RPPA.tsv": ["rppa", {}],
    # }

    # dfs = {}
    # for file in fname:
    #     if file not in omics_file_to_name_mapping:
    #         # raise Exception(f"ERROR ! '{file}' is not recognized.\n")
    #         continue
    #     if Path(omics_file_to_name_mapping[file]).exists() == False:
    #         raise Exception(f"ERROR ! File '{file}' is not found.\n")

    #     fea_name = omics_file_to_name_mapping[file][0]
    #     level_map = omics_file_to_name_mapping[file][1]
    #     header = [i for i in range(len(level_map))]
    #     fname_ = omics_file_to_name_mapping[file]

    #     df = pd.read_csv(fname_, sep=sep, index_col=0, header=header)
    #     df.index.name = canc_col_name  # assign index name
    #     df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    #     df = df.reset_index()
    #     dfs[fea_name] = df

    if omics_type == "copy_number":
        fname = "cancer_copy_number.tsv"
        level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}

    elif omics_type == "discretized_copy_number":
        fname = "cancer_discretized_copy_number.tsv"
        level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}

    elif omics_type == "methylation":
        fname = "cancer_DNA_methylation.tsv"
        level_map = {"Ensembl": 2, "Entrez": 1, "Gene_Symbol": 3, "TSS": 0}

    if omics_type == "gene_expression":
        fname = "cancer_gene_expression.tsv"
        level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}

    elif omics_type == "mirna_expression":
        # level_map = TODO
        miRNA_expression_fname = "cancer_miRNA_expression.tsv"  # cancer feature
        raise NotImplementedError(f"{omics_type} not implemeted yet.")

    elif omics_type == "mutation_count":
        fname = "cancer_mutation_count.tsv"
        level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}

    elif omics_type == "mutation":
        # level_map = TODO
        fname = "cancer_mutation.tsv"
        raise NotImplementedError(f"{omics_type} not implemeted yet.")

    elif omics_type == "rppa":
        # level_map = TODO
        fname = "cancer_RPPA.tsv"
        raise NotImplementedError(f"{omics_type} not implemeted yet.")

    else:
        raise NotImplementedError(f"Option '{omics_type}' not recognized.")

    # fpath = imp_glob.X_DATA_DIR / fname
    fpath = params["x_data_path"] / fname
    if fpath.exists() == False:
        raise Exception(f"ERROR ! {fpath} not found.\n")

    header = [i for i in range(len(level_map))]
    df = pd.read_csv(fpath, sep=sep, index_col=0, header=header)
    df.index.name = canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    df = df.reset_index()

    if verbose:
        print(f"Omics type: {omics_type}")
        print(f"Data read: {fname}")
        print(f"Shape of omics data: {df.shape}")
        print(f"Unique count of samples: {df[canc_col_name].nunique()}")
    return df


def load_and_omics(fpath: str,
                   level_map: Dict,
                   sep: str = "\t",
                   canc_col_name: str = "improve_sample_id",
                   gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
                   ):
    """ Load omics data. """
    header = [i for i in range(len(level_map))]
    df = pd.read_csv(fpath, sep=sep, index_col=0, header=header)
    df.index.name = canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    df = df.reset_index()
    return df


def load_copy_number_data(
        canc_col_name: str = "improve_sample_id",
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns copy number data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    # omics_type = "copy_number"
    fname = "cancer_copy_number.tsv"  # omics file name
    level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}

    # Create path for the omics file
    # fpath = imp_glob.X_DATA_DIR / fname
    fpath = params["x_data_path"] / fname  
    if fpath.exists() == False:
        raise Exception(f"ERROR ! {fpath} not found.\n")

    df = load_and_map_omics(fpath=fpath,
                            level_map=level_map,
                            sep=sep,
                            canc_col_name=canc_col_name,
                            gene_system_identifier=gene_system_identifier)

    if verbose:
        print(f"Copy number data: {df.shape}")
        # print(df.dtypes)
        # print(df.dtypes.value_counts())
    return df


def load_discretized_copy_number_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns discretized copy number data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # omics_type = "discretized_copy_number"
    fname = "cancer_discretized_copy_number.tsv"
    level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}

    # Create path for omics file
    fpath = params["x_data_path"] / fname  
    if fpath.exists() == False:
        raise Exception(f"ERROR ! {fpath} not found.\n")

    header = [i for i in range(len(level_map))]
    df = pd.read_csv(fpath, sep=sep, index_col=0, header=header)
    df.index.name = canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    df = df.reset_index()

    if verbose:
        print(f"Discretized copy number data: {df.shape}")
    return df


def load_dna_methylation_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns methylation data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    level_map = {"Ensembl": 2, "Entrez": 1, "Gene_Symbol": 3, "TSS": 0}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.dna_methylation_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"DNA methylation data: {df.shape}")
        # print(df.dtypes)  # TODO: many column are of type 'object'
        # print(df.dtypes.value_counts())
    return df


def load_gene_expression_data(
        canc_col_name: str = "improve_sample_id",
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns gene expression data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    # omics_type = "gene_expression"
    fname = "cancer_gene_expression.tsv"
    level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}

    # Create path for the omics file
    # fpath = imp_glob.X_DATA_DIR / fname
    fpath = params["x_data_path"] / fname  
    if fpath.exists() == False:
        raise Exception(f"ERROR ! {fpath} not found.\n")

    df = load_and_map_omics(fpath=fpath,
                            level_map=level_map,
                            sep=sep,
                            canc_col_name=canc_col_name,
                            gene_system_identifier=gene_system_identifier)

    if verbose:
        print(f"Gene expression data: {df.shape}")
    return df


def load_mirna_expression_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns miRNA expression data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # TODO
    raise NotImplementedError("The function is not implemeted yet.")
    return None


def load_mutation_count_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns mutation count data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 2, "Entrez": 0, "Gene_Symbol": 1}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.mutation_count_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"Mutation count data: {df.shape}")

    return df


def load_mutation_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns mutation data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # TODO
    raise NotImplementedError("The function is not implemeted yet.")
    return None


def load_rppa_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns mutation data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # TODO
    raise NotImplementedError("The function is not implemeted yet.")
    return None


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
        """ ... """
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

    # def load_smiles_data(self):
    #     """ ... """
    #     fpath = self.x_data_path / self.smiles_fname
    #     DrugsLoader.check_path(fpath)
    #     df = pd.read_csv(fpath, sep=self.sep)
    #     # df.columns = [self.drug_col_name, "smiles"] # Note! We updated this after updating the data
    #     # print(df.dtypes)
    #     # print(df.dtypes.value_counts())
    #     return df

    # def load_mordred_descriptor_data(self):
    #     """ ... """
    #     fpath = self.x_data_path / self.mordred_fname
    #     DrugsLoader.check_path(fpath)
    #     df = pd.read_csv(fpath, sep=self.sep)
    #     df = df.set_index(self.drug_col_name)
    #     return df

    # def load_morgan_fingerprint_data(self):
    #     """ ... """
    #     fpath = self.x_data_path / self.ecfp4_512bit_fname
    #     DrugsLoader.check_path(fpath)
    #     df = pd.read_csv(fpath, sep=self.sep)
    #     df = df.set_index(self.drug_col_name)
    #     return df

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

            # if fname == self.smiles_fname:
            #     self.dfs[fname] = self.load_smiles_data()
            # elif fname == self.mordred_fname:
            #     self.dfs[fname] = self.load_mordred_descriptor_data()
            # elif fname == self.ecfp4_512bit_fname:
            #     self.dfs[fname] = self.load_morgan_fingerprint_data()
            # else:
            #     raise NotImplementedError(f"Option '{fname}' not recognized.")

        # import ipdb; ipdb.set_trace()
        # print(self.dfs[self.smiles_fname].iloc[:4, :])
        # print(self.dfs[self.mordred_fname].iloc[:4, :4])
        # print(self.dfs[self.ecfp4_512bit_fname].iloc[:4, :4])
        # print("done")


# def load_smiles_data(
#         params: dict,
#         sep: str = "\t",
#         verbose: bool = True) -> pd.DataFrame:
#     """ Read smiles data. """
#     # fname = imp_glob.X_DATA_DIR / smiles_fname
#     fpath = params["x_data_path"] / smiles_fname
#     if fpath.exists() == False:
#         raise Exception(f"ERROR ! {fpath} not found.\n")

#     df = pd.read_csv(fpath, sep=sep)
#     df.columns = ["improve_chem_id", "smiles"] # Note! We updated this after updating the data

#     if verbose:
#         print(f"Shape of SMILES data: {df.shape}")
#         # print(df.dtypes)
#         # print(df.dtypes.value_counts())
#     return df


# def load_mordred_descriptor_data(
#         sep: str = "\t",
#         verbose: bool = True) -> pd.DataFrame:
#     """
#     Return Mordred descriptors data.
#     """
#     # df = pd.read_csv(improve_globals.mordred_file_path, sep=sep)
#     # df = df.set_index(improve_globals.drug_col_name)
#     pass
#     if verbose:
#         print(f"Mordred descriptors data: {df.shape}")
#     return df


# def load_morgan_fingerprint_data(
#         sep: str = "\t",
#         verbose: bool = True) -> pd.DataFrame:
#     """
#     Return Morgan fingerprints data.
#     """
#     # df = pd.read_csv(improve_globals.ecfp4_512bit_file_path, sep=sep)
#     # df = df.set_index(improve_globals.drug_col_name)
#     pass
#     return df


# def load_drug_data(fname: Union[Path, str],
def load_drug_data(fname: Union[str, Path, List[str], List[Path]] = ["drug_SMILES.tsv"],
                   index: Optional[str] = None,
                   columns: Optional[List] = None,
                   sep: str = "\t",
                   verbose: bool = True) -> pd.DataFrame:
    """
    Returns dataframe with drug features read from specified file.
    Index or columns may be redefined if requested.

    Args:
        fname: Name of, or Path to, file for reading drug data.
        index: Name of column to set as index in the data frame read.
        columns: List to rename data frame columns. Default: None, i.e. do not rename columns.
        sep: Separator used in data file.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
    """
    # Drug features file names
    # smiles_file_name = "drug_SMILES.tsv"  # drug feature
    # mordred_file_name = "drug_mordred.tsv"  # drug feature
    # ecfp4_512bit_file_name = "drug_ecfp4_512bit.tsv"  # drug feature

    drug_file_to_func_mapping = {
        "drug_SMILES.tsv": ["smiles", load_smiles_data],
        "drug_mordred.tsv": ["mordred", load_mordred_descriptor_data],
        "drug_ecfp4_nbits512.tsv": ["fps", load_morgan_fingerprint_data],
    }

    dfs = {}
    for file in fname:
        if file not in drug_file_to_func_mapping:
            # raise Exception(f"ERROR ! '{file}' is not recognized.\n")
            continue
        df = drug_file_to_func_mapping[f]()
        dfs[fea_name] = df

    # df = pd.read_csv(fname, sep=sep)

    # if columns is not None: # Set columns
    #     df.columns = columns

    # if index is not None: # Set index
    #     df = df.set_index(index)

    if verbose:
        print(f"Data read: {fname}")
        print(f"Shape of constructed drug data framework: {df.shape}")
    return df


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
        """ ... """
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


# moved from graphdrp_preprocess_improve.py
def load_response_data(
        y_data_fpath,
        # inpath_dict: frm.DataPathDict,
        # y_file_name: str,
        source: str,
        # split_id: int,
        # split_file_name: Union[str, List[str], None] = None,
        split_fpath: Union[str, Path],
        # stage: str,
        canc_col_name: str = "improve_sample_id",
        drug_col_name: str = "improve_chem_id",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns dataframe with cancer ids, drug ids, and drug response values.
    Samples from the original drug response file are filtered based on
    the specified split ids.

    :params: Dict inpath_dict: Dictionary of paths and info about raw
             data input directories.
    :params: str y_file_name: Name of file for reading the y_data.
    :params: str source: DRP source name.
    :params: int split_id: Split id. If -1, use all data. Note that this
             assumes that split_id has been constructed to take into
             account all the data sources.
    :params: str stage: Type of partition to read. One of the following:
             'train', 'val', 'test'.
    :params: str canc_col_name: Column name that contains the cancer
             sample ids. Default: "improve_sample_id".
    :params: str drug_col_name: Column name that contains the drug ids.
             Default: "improve_chem_id".
    :params: str sep: Separator used in data file.
    :params: bool verbose: Flag for verbosity. If True, info about
             computations is displayed. Default: True.

    :return: Dataframe that contains single drug response values.
    :rtype: pd.Dataframe
    """
    # y_data_file = inpath_dict["y_data"] / y_file_name
    if y_data_fpath.exists() == False:
        raise Exception(f"ERROR ! {y_data_fpath} file not available.\n")
    # Read y_data_file
    df = pd.read_csv(y_data_fpath, sep=sep)

    # # Get a subset of samples if split_id is different to -1
    # if split_id > -1:
    #     # TODO: this should not be encoded like this because other comparison
    #     # piplines will have a different split_file_name.
    #     # E.g, in learning curve, it will be
    #     # f"{source}_split_{split_id}_{stage}_size_{train_size}.txt"
    #     # Moreover, we should be able to pass a list of splits.
    #     split_file_name = f"{source}_split_{split_id}_{stage}.txt"
    # else:
    #     split_file_name = f"{source}_all.txt"

    # insplit = inpath_dict["splits"] / split_file_name
    # if insplit.exists() == False:
    #     raise Exception(f"ERROR ! {split_file_name} file not available.\n")

    if split_fpath.exists() == False:
        raise Exception(f"ERROR ! {split_fpath} not found.\n")

    ids = pd.read_csv(split_fpath, header=None)[0].tolist()
    df = df.loc[ids]

    df = df.reset_index(drop=True)
    if verbose:
        print(f"Data read: {y_data_fpath}, Filtered by: {split_fpath}")
        print(f"Shape of constructed response data framework: {df.shape}")
        print(f"Unique cells:  {df[canc_col_name].nunique()}")
        print(f"Unique drugs:  {df[drug_col_name].nunique()}")
    return df

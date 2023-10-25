"""Functionality for IMPROVE data handling."""

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler


# level_map_cell_data encodes the relationship btw the column and gene identifier system
level_map_cell_data = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}


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


def common_elements(list1: List, list2: List, verbose: bool = True) -> List:
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


def load_cell_data(fname: Union[Path, str],
        canc_col_name = "improve_sample_id",
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns data frame with specified cell line data.

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
    header = [i for i in range(len(level_map_cell_data))]

    df = pd.read_csv(fname, sep=sep, index_col=0, header=header)
    df.index.name = canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map_cell_data, gene_system_identifier)
    df = df.reset_index()
    if verbose:
        print(f"Data read: {fname}")
        print(f"Shape of constructed cell data framework: {df.shape}")
        print(f"Unique cells:  {df[canc_col_name].nunique()}")
    return df


def load_drug_data(fname: Union[Path, str],
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
    df = pd.read_csv(fname, sep=sep)

    if columns is not None: # Set columns
        df.columns = columns

    if index is not None: # Set index
        df = df.set_index(index)

    if verbose:
        print(f"Data read: {fname}")
        print(f"Shape of constructed drug data framework: {df.shape}")
    return df


def scale_df(dataf, scaler_name="std", scaler = None, verbose=False):
    """ Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        dataf: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return dataf, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = dataf.select_dtypes(include="number")

    if scaler is None: # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(f"The specified scaler {scaler_name} is not implemented (no df scaling).")
            return dataf, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else: # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    dataf[df_num.columns] = df_norm
    return dataf, scaler

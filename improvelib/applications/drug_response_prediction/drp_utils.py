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


def common_elements(list1: List, list2: List, verbose: bool = False) -> List:
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

        self.params = params
        self.sep = sep
        if isinstance(params["y_data_files"], str):
            # instanciate array from string
            self.inp = literal_eval(params["y_data_files"])
        else:
            self.inp = params["y_data_files"]

        self.y_col_name = params["y_col_name"]
        self.canc_col_name = params["canc_col_name"]
        self.drug_col_name = params["drug_col_name"]

        # self.y_data_path = params["y_data_path"]/params["y_data_files"][0][0]
        self.y_data_path = params["y_data_path"]
        self.split_fpath = Path(params["splits_path"]) / split_file
        self.dfs = {}
        self.verbose = verbose

        if self.verbose:
            print(f"y_data_files: {params['y_data_files']}")
            print(f"y_col_name: {params['y_col_name']}")

        # self.inp_fnames = []
        # for i in self.inp:
        #     logger.debug(f"i: {i}")
        #     # was [['response.tsv']] but now ['response.tsv']
        #     # assert len(i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
        #     self.inp_fnames.append(i)
        # logger.debug(self.inp_fnames)

        self.load_all_response_data()

    def __repr__(self):
        if self.dfs:
            to_print = []
            to_print.append("Loaded data:\n")
            to_print.append("\n".join(
                [f"{fname}: {df.shape} \nUnique cells: {df[self.canc_col_name].nunique()} \nUnique drugs: {df[self.drug_col_name].nunique()}" for fname, df in self.dfs.items()]))
            to_print = "".join(to_print)
            return to_print
        else:
            return "No data files were loaded."

    @ staticmethod
    def check_path(fpath):
        fpath = Path(fpath)
        if fpath.exists() == False:
            raise Exception(f"ERROR ! {fpath} not found.\n")

    def load_response_data(self, fname):
        fpath = Path(os.path.join(str(self.y_data_path), fname))
        logger.debug(f"Loading {fpath}")
        DrugResponseLoader.check_path(fpath)
        df = pd.read_csv(fpath, sep=self.sep)
        return df

    def load_all_response_data(self, verbose: str = True):
        """ ... """
        for i in self.inp[0]:
            fname = i
            df = self.load_response_data(fname)
            DrugResponseLoader.check_path(self.split_fpath)
            ids = pd.read_csv(self.split_fpath, header=None)[0].tolist()
            df = df.loc[ids]
            self.dfs[fname] = df

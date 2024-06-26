""" Functionality for IMPROVE drug response prediction (DRP) models. """

from . import framework as frm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import sys
import numpy as np

from ast import literal_eval

import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum


# Set logger for this module

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))


# from improve import framework as frm
# from .framework import imp_glob  ## ap; removed

# ---------------------------------------------------------------
# File names that should be accesible to all IMPROVE DRP models.
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
                level_map[gene_system_identifier])  # retain specific column level
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

        self.params = params
        self.sep = sep

        if isinstance(params["x_data_drug_files"], str):
            # instanciate array from string
            self.inp = literal_eval(params["x_data_drug_files"])
        else:
            self.inp = params["x_data_drug_files"]

        logger.debug(f"self.inp: {self.inp}")

        self.drug_col_name = params["drug_col_name"]
        self.x_data_path = params["x_data_path"]
        self.dfs = {}
        self.verbose = verbose

        if self.verbose:
            # print(f"x_data_drug_files: {params['x_data_drug_files']}")
            print(f"drug_col_name: {params['drug_col_name']}")
            print("x_data_drug_files:")
            for i, d in enumerate(self.inp):
                print(f"{i+1}. {d}")

        self.inp_fnames = []
        for i in self.inp:
            assert len(
                i) == 1, f"Inner lists must contain only one item, but {i} is {len(i)}"
            self.inp_fnames.append(i[0])
        # print(self.inp_fnames)

        self.load_all_drug_data()

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

    def load_drug_data(self, fname):
        """ ... """
        fpath = os.path.join(self.x_data_path, fname)
        DrugsLoader.check_path(fpath)
        # if self.verbose:
        #     print(f"Loading {fpath}")
        df = pd.read_csv(fpath, sep=self.sep)
        df = df.set_index(self.drug_col_name)
        return df

    def load_all_drug_data(self):
        """ ... """
        for i in self.inp:
            fname = i[0]
            self.dfs[fname] = self.load_drug_data(fname)

        # print(self.dfs[self.smiles_fname].iloc[:4, :])
        # print(self.dfs[self.mordred_fname].iloc[:4, :4])
        # print(self.dfs[self.ecfp4_512bit_fname].iloc[:4, :4])
        print("Finished loading drug data.")


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
        self.split_fpath = os.path.join(params["splits_path"], split_file)
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
        fpath = Path(os.path.join(self.y_data_path, fname))
        logger.debug(f"Loading {fpath}")
        DrugResponseLoader.check_path(fpath)
        df = pd.read_csv(fpath, sep=self.sep)
        return df

    def load_all_response_data(self, verbose: str = True):
        """ ... """
        for i in self.inp:
            fname = i
            df = self.load_response_data(fname)
            DrugResponseLoader.check_path(self.split_fpath)
            ids = pd.read_csv(self.split_fpath, header=None)[0].tolist()
            df = df.loc[ids]
            self.dfs[fname] = df


class Benchmark(ABC):

    def set_benchmark_dir(self, benchmark_dir: str):
        self._benchmark_dir = benchmark_dir

    @ abstractmethod
    def set_dataset(self, dataset: Enum) -> None:
        pass

    @ abstractmethod
    def get_dataframe(self, dataframe: Enum) -> pd.DataFrame:
        pass


class StringEnum(Enum):
    def __str__(self):
        return str(self.value)


class SingleDRPDataFrame(StringEnum):
    CELL_LINE_CNV = 'cnv'
    CELL_LINE_DISCRETIZED_CNV = 'cnv_discretized'
    CELL_LINE_METHYLATION = 'methylation'
    CELL_LINE_GENE_EXPRESSION = 'gene_expression'
    CELL_LINE_miRNA = 'miRNA'
    CELL_LINE_MUTATION_COUNT = 'mutation_count'
    CELL_LINE_MUTATION_LONG_FORMAT = 'mutation_long_format'
    CELL_LINE_MUTATION = 'mutation'
    CELL_LINE_RPPA = 'rppa'
    DRUG_MORDRED = 'mordred'
    DRUG_SMILES = 'smiles'
    DRUG_ECFP4_NBITS512 = 'ecfp_nbits512'
    RESPONSE = 'response'


class SingleDRPDataset(StringEnum):
    CCLE = 'CCLE'
    CTRP = 'CTRPv2'
    GDSCv1 = 'GDSCv1'
    GDSCv2 = 'GDSCv2'
    gCSI = 'gCSI'


class SplitType(StringEnum):
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'


class DRPMetric(StringEnum):
    AUC = 'auc'
    AUC1 = 'auc1'
    IC50 = 'ic50'


class SingleDRPBenchmark(Benchmark):
    CANCER_COL_NAME = "improve_sample_id"
    DRUG_COL_NAME = "improve_chem_id"

    # Setting required specifications for generating output data frames
    def set_dataset(self, dataset: SingleDRPDataset) -> None:
        self._dataset = dataset

    def set_split_id(self, split_number: int) -> None:
        self._split_id = split_number

    def set_split_type(self, split_type: SplitType) -> None:
        self._split_type = split_type

    def set_drp_metric(self, metric: DRPMetric) -> None:
        self._drp_metric = metric

    # Optional parameters

    def set_splits_dir(self, splits_dir: str) -> None:
        """
        Setting splits dir is required only if new splits directory is provided.
        New splits directory should be located in the same parent directory as
        the default directory of SingleDRPBenchmark.
        """
        self._splits_dir = splits_dir

    # Getting output from
    def get_splits_num(self) -> int:
        return self._SPLITS_NUM

    def get_dataframe(self, dataframe: SingleDRPDataFrame) -> pd.DataFrame:
        return self._load_dataframe(dataframe)

    def get_full_dataframe(self, dataframe: SingleDRPDataFrame) -> pd.DataFrame:
        prev_split_id = self._split_id
        self._split_id = 'all'
        df = self.get_dataframe(dataframe)
        self._split_id = prev_split_id
        return df

    def __init__(self):
        self._SPLITS_NUM = 10
        self._benchmark_dir = None
        self._dataset = None
        self._split_id = None
        self._split_type = None
        self._drp_metric = None
        self._loaded_dfs = {}
        self._splits_dir = 'splits'
        self._dataset2file_map = {
            SingleDRPDataFrame.CELL_LINE_CNV: "cancer_copy_number.tsv",
            SingleDRPDataFrame.CELL_LINE_DISCRETIZED_CNV: "cancer_discretized_copy_number.tsv",
            SingleDRPDataFrame.CELL_LINE_METHYLATION: "cancer_DNA_methylation.tsv",
            SingleDRPDataFrame.CELL_LINE_GENE_EXPRESSION: "cancer_gene_expression.tsv",
            SingleDRPDataFrame.CELL_LINE_miRNA: "cancer_miRNA_expression.tsv",
            SingleDRPDataFrame.CELL_LINE_MUTATION_COUNT: "cancer_mutation_count.tsv",
            SingleDRPDataFrame.CELL_LINE_MUTATION_LONG_FORMAT: "cancer_mutation_long_format.tsv",
            SingleDRPDataFrame.CELL_LINE_MUTATION: "cancer_mutation.parquet",
            SingleDRPDataFrame.CELL_LINE_RPPA: "cancer_RPPA.tsv",
            SingleDRPDataFrame.DRUG_SMILES: "drug_SMILES.tsv",
            SingleDRPDataFrame.DRUG_MORDRED: "drug_mordred.tsv",
            SingleDRPDataFrame.DRUG_ECFP4_NBITS512: "drug_ecfp4_nbits512.tsv",
            SingleDRPDataFrame.RESPONSE: "response.tsv"
        }

    def _check_initialization(self) -> None:
        template_message = "is not specified in benchmark!"
        if self._benchmark_dir is None:
            raise Exception(f"Dataset {template_message}")
        if self._dataset is None:
            raise Exception(f"Dataset {template_message}")
        if self._split_id is None:
            raise Exception(f"Split ID {template_message}")
        if self._split_type is None:
            raise Exception(f"Split type {template_message}")
        if self._drp_metric is None:
            raise Exception(f"DRP Metric {template_message}")

    def _construct_splits_file_name(self) -> str:
        if self._split_id == 'all':
            fname = '_'.join((str(self._dataset), 'all'))
            return f'{fname}.txt'
        filename = '_'.join((str(self._dataset), 'split',
                             str(self._split_id), str(self._split_type)))
        filename = f'{filename}.txt'
        return filename

    def _load_cancer_dataframe(self, dataframe: SingleDRPDataFrame) -> pd.DataFrame:
        dataframe_file = self._dataset2file_map[dataframe]
        loader_params = {}
        loader_params["x_data_canc_files"] = str([[dataframe_file]])
        loader_params["canc_col_name"] = self.CANCER_COL_NAME
        loader_params["x_data_path"] = os.path.join(
            self._benchmark_dir, 'x_data')
        omics_loader = OmicsLoader(loader_params)
        return omics_loader.dfs[dataframe_file]

    def _load_drug_dataframe(self, dataframe: SingleDRPDataFrame) -> pd.DataFrame:
        dataframe_file = self._dataset2file_map[dataframe]
        loader_params = {}
        loader_params["x_data_drug_files"] = str([[dataframe_file]])
        loader_params["drug_col_name"] = self.DRUG_COL_NAME
        loader_params["x_data_path"] = os.path.join(
            self._benchmark_dir, 'x_data')
        drug_loader = DrugsLoader(loader_params)
        return drug_loader.dfs[dataframe_file]

    def _load_response_dataframe(self, dataframe: SingleDRPDataFrame) -> pd.DataFrame:
        dataframe_file = self._dataset2file_map[dataframe]
        loader_params = {}
        loader_params["y_data_files"] = str([dataframe_file])
        loader_params["canc_col_name"] = self.CANCER_COL_NAME
        loader_params["drug_col_name"] = self.DRUG_COL_NAME
        loader_params["y_col_name"] = str(self._drp_metric)
        loader_params["y_data_path"] = os.path.join(
            self._benchmark_dir, 'y_data')
        loader_params["splits_path"] = os.path.join(
            self._benchmark_dir, self._splits_dir)
        split_file = self._construct_splits_file_name()
        response_loader = DrugResponseLoader(
            loader_params, split_file=split_file)
        return response_loader.dfs[dataframe_file]

    def _load_dataframe(self, dataframe: SingleDRPDataFrame) -> pd.DataFrame:
        self._check_initialization()
        df = None
        if dataframe in self._loaded_dfs:
            df = self._loaded_dfs[dataframe]
        else:
            if dataframe not in self._dataset2file_map:
                raise Exception(
                    f'Dataframe name {dataframe} is not mapped to the file!')

            ids = None
            if 'cancer' in self._dataset2file_map[dataframe]:
                df = self._load_cancer_dataframe(dataframe)
            elif 'drug' in self._dataset2file_map[dataframe]:
                df = self._load_drug_dataframe(dataframe)
            elif 'response' in self._dataset2file_map[dataframe]:
                return self._load_response_dataframe(dataframe)
            else:
                raise Exception(
                    f'Dataframe name {dataframe} is mapped to the unknown file name')
            self._loaded_dfs[dataframe] = df

        key_col_name = None
        if 'cancer' in self._dataset2file_map[dataframe]:
            key_col_name = self.CANCER_COL_NAME
        elif 'drug' in self._dataset2file_map[dataframe]:
            key_col_name = self.DRUG_COL_NAME

        response_df = self._load_response_dataframe(
            SingleDRPDataFrame.RESPONSE)
        ids = np.unique(response_df[key_col_name])

        index_name = df.index.name
        index_name = 'index' if index_name is None else index_name
        df_split = df.reset_index(drop=False)
        df_split.set_index(key_col_name, drop=False, inplace=True)
        df_split = df_split.loc[ids]
        df_split.set_index(index_name, inplace=True, drop=True)

        return df_split


class SingleDRPDataStager():

    def __init__(self) -> None:
        self._out_file_name = 'data.hdf5'
        self._out_dir = None
        self._benchmark = None

    def set_benchmark(self, benchmark):
        self._benchmark = benchmark

    def set_output_dir(self, output_dir: str) -> None:
        self._out_dir = output_dir

    # def stage_all_experiments(self,
    #                           single_drp_datasets: list[SingleDRPDataset],
    #                           data_frame_list: list[SingleDRPDataFrame],
    #                           drp_metric: DRPMetric) -> dict[dict[dict[str]]]: # python >3.9
    def stage_all_experiments(self,
                              single_drp_datasets: List[SingleDRPDataset],
                              data_frame_list: List[SingleDRPDataFrame],
                              drp_metric: DRPMetric) -> Dict[str, Dict[str, Dict[str, str]]]:
        self._check_initialization()
        path_dict = {}
        self._benchmark.set_drp_metric(drp_metric)
        splits_num = self._benchmark.get_splits_num()
        for dataset in single_drp_datasets:
            path_dict[dataset] = {}
            self._benchmark.set_dataset(dataset)
            for split_id in range(splits_num):
                path_dict[dataset][split_id] = {}
                self._benchmark.set_split_id(split_id)
                for split_type in [SplitType.TRAIN, SplitType.VALIDATION, SplitType.TEST]:
                    self._benchmark.set_split_type(split_type)

                    sub_dir = self._construct_out_sub_dir(
                        dataset, split_id, split_type)

                    # Stub for saving data in fixed format
                    out_file_path = os.path.join(sub_dir, self._out_file_name)
                    path_dict[dataset][split_id][split_type] = out_file_path
                    for dataframe_name in data_frame_list:
                        dataframe = self._benchmark.get_dataframe(
                            dataframe_name)
                        # dataframe.to_hdf(str(dataframe_name),
                        #                 out_file_path, mode='a')
        return path_dict

    def _check_initialization(self) -> None:
        template_message = "is not initialized!"
        if self._out_dir is None:
            raise Exception(f"Output dir for staging {template_message}")
        if self._benchmark is None:
            raise Exception(f"Benchmark for data staging {template_message}")

    def _construct_out_sub_dir(self, single_drp_dataset: SingleDRPDataset, split_id: int, split_type: SplitType) -> str:
        path = os.path.join(self._out_dir, str(single_drp_dataset),
                            f'split_{str(split_id)}', str(split_type))
        frm.create_outdir(path)
        return path

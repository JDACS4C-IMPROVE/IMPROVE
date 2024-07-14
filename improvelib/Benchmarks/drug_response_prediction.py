from improvelib.benchmarks import base as Base
from improvelib.benchmarks.base import Benchmark, Stage, ParameterConverter, DatasetDescription, DataFrameDescription
from enum import Enum

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

import pandas as pd
from improvelib.applications.drug_response_prediction import drp_utils
from improvelib.applications.drug_response_prediction import omics_utils
from improvelib.applications.drug_response_prediction import drug_utils


class DRP(Base.Base):
    """Class to handle configuration files for Drug Response Prediction."""
    # Set section for config file
    section = 'DRPBenchmark_v1.0'

    # Default format for logging
    FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('DRP')
    # logger=logging.getLogger(__name__)
    logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", logging.ERROR))

    # Set options for command line
    drp_options = [
        {
            'name': 'benchmark_dir',
            'default': './',
            'type': str,
            'help': 'Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.'
        },
        {
            'name': 'training_index_file',
            'default': 'training.idx',
            'type': str,
            'help': 'index file for training set [numpy array]'
        },
        {
            'name': 'validation_index_file',
            'default': 'validation.idx',
            'type': str,
            'help': 'index file for validation set [numpy array]',
        },
        {
            'name': 'testing_index_file',
            'default': 'testing.idx',
            'type': str,
            'help': 'index file for testiing set [numpy array]',
        },
        {
            'name': 'data',
            'default': 'data.parquet',
            'type': str,
            'help': 'data file',
        },
        {
            'name': 'input_type',
            'default': 'BenchmarkV1',
            'choices': ['parquet', 'csv', 'hdf5', 'npy', 'BenchmarkV1'],
            'metavar': 'TYPE',
            'help': 'Sets the input type. Default is BenchmarkV1. Other options are parquet, csv, hdf5, npy'
        }
    ]

    def __init__(self) -> None:
        super().__init__()
        self.logger = DRP.logger
        self.options = DRP.drp_options
        self.input_dir = None
        self.x_data_path = None
        self.y_data_path = None
        self.splits_path = None

    def init(self, cfg, verbose=False):
        """Initialize Drug Response Prediction Benchmark. Takes Config object as input."""

        if cfg.log_level:
            self.logger.setLevel(cfg.log_level)
            self.logger.debug(f"Log level set to {cfg.log_level}.")
        else:
            self.logger.warning(
                "No log level set in Config object. Using default log level.")

        self.logger.debug("Initializing Drug Response Prediction Benchmark.")
        self.set_input_dir(cfg.get_param('input_dir'))
        self.set_output_dir(cfg.get_param('output_dir'))

        self.params = cfg.dict()

        self.dfs = {}
        self.verbose = verbose
        if self.verbose:
            print(f"y_data_files: {self.params['y_data_files']}")
            print(f"y_col_name: {self.params['y_col_name']}")

        self.inp_fnames = []

    def set_input_dir(self, input_dir: str) -> None:
        """Set input directory for Drug Response Prediction Benchmark."""

        # check if input_dir is Path object otherwise convert to Path object
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)

        self.input_dir = input_dir
        self.x_data_path = Path(input_dir) / "x_data"
        self.y_data_path = Path(input_dir) / "y_data"
        self.splits_path = Path(input_dir) / "splits"


DRUG_KEY_COL = 'improve_chem_id'
CANCER_KEY_COL = 'improve_sample_id'


class SingleDRPDataFrame(Enum):
    """
    Enum for specifying single drug response prediction dataframes.
    """

    CELL_LINE_CNV = DataFrameDescription(name='cnv',
                                         file='cancer_copy_number.tsv',
                                         key_column=CANCER_KEY_COL,
                                         group='cancer',
                                         description='Copy number variation')
    CELL_LINE_DISCRETIZED_CNV = DataFrameDescription(name='cnv_discretized',
                                                     file='cancer_discretized_copy_number.tsv',
                                                     key_column=CANCER_KEY_COL,
                                                     group='cancer',
                                                     description='Discretized copy number variation')
    CELL_LINE_METHYLATION = DataFrameDescription(name='methylation',
                                                 file='cancer_DNA_methylation.tsv',
                                                 key_column=CANCER_KEY_COL,
                                                 group='cancer',
                                                 description='DNA methylation data for the cell lines')
    CELL_LINE_GENE_EXPRESSION = DataFrameDescription(name='gene_expression',
                                                     file='cancer_gene_expression.tsv',
                                                     key_column=CANCER_KEY_COL,
                                                     group='cancer',
                                                     description='Gene expression data for the cell lines')
    CELL_LINE_miRNA = DataFrameDescription(name='miRNA',
                                           file='cancer_miRNA_expression.tsv',
                                           key_column=CANCER_KEY_COL,
                                           group='cancer',
                                           description='Micro RNA (miRNA) data for the cell lines')
    CELL_LINE_MUTATION_COUNT = DataFrameDescription(name='mutation_count',
                                                    file='cancer_mutation_count.tsv',
                                                    key_column=CANCER_KEY_COL,
                                                    group='cancer',
                                                    description='Mutation count data for the cell lines')
    CELL_LINE_MUTATION_LONG_FORMAT = DataFrameDescription(name='mutation_long_format',
                                                          file='cancer_mutation_long_format.tsv',
                                                          key_column=CANCER_KEY_COL,
                                                          group='cancer',
                                                          description='Full data on genetic mutations for the cell lines')
    CELL_LINE_MUTATION = DataFrameDescription(name='mutation',
                                              file='cancer_mutation.parquet',
                                              key_column=CANCER_KEY_COL,
                                              group='cancer',
                                              description='Shortened data on mutation for the cell lines')
    CELL_LINE_RPPA = DataFrameDescription(name='rppa',
                                          file='cancer_RPPA.tsv',
                                          key_column=CANCER_KEY_COL,
                                          group='cancer',
                                          description='Reverse-phase protein array data for the cell lines')
    DRUG_MORDRED = DataFrameDescription(name='mordred',
                                        file='drug_mordred.tsv',
                                        key_column=DRUG_KEY_COL,
                                        group='drug',
                                        description='MORDRED computational descriptors for the drug data')
    DRUG_SMILES = DataFrameDescription(name='smiles',
                                       file='drug_SMILES.tsv',
                                       key_column=DRUG_KEY_COL,
                                       group='drug',
                                       description='SMILES representation of the drug compound')
    DRUG_ECFP4_NBITS512 = DataFrameDescription(name='ecfp4_nbits512',
                                               file='drug_ecfp4_nbits512.tsv',
                                               key_column=DRUG_KEY_COL,
                                               group='drug',
                                               description='Extended Connectivity Fingerprints V4')
    RESPONSE = DataFrameDescription(name='response',
                                    file='response.tsv',
                                    key_column=None,
                                    group='response',
                                    description='Drug response data for the IMPROVE project. The entry is uniquely defined by improve_sample_id and improve_drug_id columns')


class SingleDRPDataset(Enum):
    """
    Enum for specifying datasets in single drug response prediction benchmarks.
    """
    CCLE = DatasetDescription(name='CCLE',
                              description='Cancer Cell Line Encyclopedia')
    CTRP = DatasetDescription(name='CTRP',
                              description='Cancer Therapeutic Response Portal')
    GDSCv1 = DatasetDescription(name='GDSCv1',
                                description='Genomics of Drug Sensitivity in Cancer V1')
    GDSCv2 = DatasetDescription(name='GDSCv2',
                                description='Genomics of Drug Sensitivity in Cancer V2')
    gCSI = DatasetDescription(name='gCSI',
                              description='The Genentech Cell Line Screening')


class SingleDRPMetric(Enum):
    """
    Enum for specifying the drug response prediction metrics.
    """
    AUC = 'auc'
    AUC1 = 'auc1'
    IC50 = 'ic50'
    EC50 = 'ec50'
    EC50se = 'ec50se'
    R2FIT = 'r2fit'
    EINF = 'einf'
    HS = 'hs'
    AAC1 = 'aac1'
    DSS1 = 'dss1'


class SingleDRPParameterConverter(ParameterConverter):
    def __init__(self):
        super().__init__(
            [SingleDRPDataFrame, SingleDRPDataset, Stage, SingleDRPMetric])


class SingleDRPBenchmark(Benchmark):
    """
    Concrete implementation of the Benchmark class for single drug response prediction.

    Attributes:
        CANCER_COL_NAME (str): Default column name for cancer identifiers.
        DRUG_COL_NAME (str): Default column name for drug identifiers.

    Methods:
        set_dataset(dataset: SingleDRPDataset): Sets the dataset for the benchmark.
        set_split_id(split_number: int): Sets the split number for data partitioning.
        set_stage(stage: Stage): Sets the type of data split.
        set_metric(metric: DRPMetric): Sets the drug response prediction metric.
        set_splits_dir(splits_dir: str): Sets the directory for data splits.
        get_splits_ids() -> list[int]: Returns splits IDs.
        get_dataframe(dataframe: SingleDRPDataFrame) -> pd.DataFrame: Retrieves a dataframe based on the specified parameters.
        get_full_dataframe(dataframe: SingleDRPDataFrame) -> pd.DataFrame: Retrieves a full dataframe without considering splits.
    """

    def __init__(self):
        """
        Initializes the SingleDRPBenchmark instance with default values.
        """
        self._initialize()
        self._SPLITS_NUM = 10
        self._splits_ids = list(range(self._SPLITS_NUM))
        self._loaded_dfs = {}
        self._splits_dir = 'splits'

    def get_datasets(self):
        return SingleDRPDataset

    def get_dataframes(self):
        return SingleDRPDataFrame

    def get_stages(self):
        return Stage

    def get_metrics(self):
        return SingleDRPMetric

    # Setting required specifications for generating output data frames
    def set_dataset(self, dataset: SingleDRPDataset):
        """
        Sets the dataset for the benchmark.

        Args:
            dataset (SingleDRPDataset): The dataset to be used in the benchmark.x
        """
        self._dataset = dataset

    def set_metric(self, metric: SingleDRPMetric):
        """
        Sets the drug response prediction metric.

        Args:
            metric (DRPMetric): The metric to be used for evaluating drug response.
        """
        self._metric = metric

    # Getting data from benchmark
    def get_splits_ids(self):
        """
        Returns the number of data splits.

        Returns:
            int: The number of data splits.
        """
        return self._splits_ids

    def get_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Retrieves a dataframe based on the specified parameters.

        Args:
            dataframe (SingleDRPDataFrame): The type of dataframe to retrieve.

        Returns:
            pd.DataFrame: The requested dataframe.
        """
        return self._load_dataframe(dataframe)

    def get_full_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Retrieves a full dataframe without considering splits.

        Args:
            dataframe (SingleDRPDataFrame): The type of dataframe to retrieve.

        Returns:
            pd.DataFrame: The full dataframe.
        """
        prev_split_id = self._split_id
        prev_stage = self._stage
        self._split_id = 'all'
        self._stage = ''
        df = self.get_dataframe(dataframe)
        self._split_id = prev_split_id
        self._stage = prev_stage
        return df

    def get_metric(self):
        return self._metric

    def _construct_splits_file_name(self):
        """
        Constructs the file name for the data splits based on the current dataset and split settings.

        Returns:
            str: The constructed file name for the splits.
        """
        if self._split_id == 'all':
            fname = '_'.join((str(self._dataset.value.name), 'all'))
            return f'{fname}.txt'
        filename = '_'.join((str(self._dataset.value.name), 'split',
                             str(self._split_id), str(self._stage.value)))
        filename = f'{filename}.txt'
        return filename

    def _load_cancer_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Loads a cancer-related dataframe based on the specified dataframe type.

        Args:
            dataframe (SingleDRPDataFrame): The type of cancer-related dataframe to load.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        dataframe_file = dataframe.value.file
        loader_params = {}
        loader_params["x_data_canc_files"] = str([[dataframe_file]])
        loader_params["canc_col_name"] = dataframe.value.key_column
        loader_params["x_data_path"] = os.path.join(
            self._benchmark_dir, 'x_data')
        omics_loader = omics_utils.OmicsLoader(loader_params)
        return omics_loader.dfs[dataframe_file]

    def _load_drug_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Loads a drug-related dataframe based on the specified dataframe type.

        Args:
            dataframe (SingleDRPDataFrame): The type of drug-related dataframe to load.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        dataframe_file = dataframe.value.file
        loader_params = {}
        loader_params["x_data_drug_files"] = str([[dataframe_file]])
        loader_params["drug_col_name"] = dataframe.value.key_column
        loader_params["x_data_path"] = os.path.join(
            self._benchmark_dir, 'x_data')
        drug_loader = drug_utils.DrugsLoader(loader_params)
        return drug_loader.dfs[dataframe_file]

    def _load_response_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Loads a response dataframe based on the specified dataframe type.

        Args:
            dataframe (SingleDRPDataFrame): The type of response dataframe to load.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        dataframe_file = dataframe.value.file
        loader_params = {}
        loader_params["y_data_files"] = str([[dataframe_file]])
        loader_params["canc_col_name"] = CANCER_KEY_COL
        loader_params["drug_col_name"] = DRUG_KEY_COL
        loader_params["y_col_name"] = str(self._metric.value)
        loader_params["y_data_path"] = os.path.join(
            self._benchmark_dir, 'y_data')
        loader_params["splits_path"] = os.path.join(
            self._benchmark_dir, self._splits_dir)
        split_file = self._construct_splits_file_name()
        response_loader = drp_utils.DrugResponseLoader(
            loader_params, split_file=split_file)
        df = response_loader.dfs[dataframe_file]
        cols_to_drop = [col for col in df.columns if col not in [
            CANCER_KEY_COL, DRUG_KEY_COL, str(self._metric.value)]]
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    def _load_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Loads a dataframe based on the specified dataframe type, ensuring it is initialized and filtered by relevant IDs.

        Args:
            dataframe (SingleDRPDataFrame): The type of dataframe to load.

        Returns:
            pd.DataFrame: The loaded and filtered dataframe.
        """
        self._check_initialization()
        df = None
        if dataframe in self._loaded_dfs:
            df = self._loaded_dfs[dataframe]
        else:
            if dataframe.value.group == 'cancer':
                df = self._load_cancer_dataframe(dataframe)
            elif dataframe.value.group == 'drug':
                df = self._load_drug_dataframe(dataframe)
            elif dataframe.value.group == 'response':
                return self._load_response_dataframe(dataframe)
            else:
                raise Exception(
                    f'Dataframe {dataframe.value.name} belongs to unknown group! Cannot proceed with data loading')
            self._loaded_dfs[dataframe] = df

        key_col_name = dataframe.value.key_column
        response_df = self._load_response_dataframe(
            SingleDRPDataFrame.RESPONSE)
        ids = response_df[key_col_name].unique()

        index_name = df.index.name
        index_name = 'index' if index_name is None else index_name
        df_split = df.reset_index(drop=False)
        df_split.set_index(key_col_name, drop=False, inplace=True)
        df_split = df_split.loc[ids]
        df_split.set_index(index_name, inplace=True, drop=True)

        return df_split


if __name__ == "__main__":
    drp = DRP()
    drp.set_input_dir("input_dir")
    print(drp.__dict__)

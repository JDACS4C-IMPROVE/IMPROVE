from improvelib.Benchmarks import Base as Base
from improvelib.Benchmarks.Base import Benchmark, Stage, ParameterConverter
from improvelib.Benchmarks.benchmark_utils import StringEnum

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

import pandas as pd
import improvelib.drug_resp_pred as drp


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

    def build_paths(self, config: None, params: Dict):
        """ Build paths for raw_data, x_data, y_data, splits.
        These paths determine directories for a benchmark dataset.
        TODO: consider renaming to build_benchmark_data_paths()

        Args:
            params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

        Returns:
            dict: updated dict of CANDLE/IMPROVE parameters and parsed values.
        """
        # mainpath = self.input_dir
        check_path(self.input_dir)

        # Raw data
        raw_data_path = mainpath / params["raw_data_dir"]
        config.set_param("raw_data_path", raw_data_path)

        # params["raw_data_path"] = raw_data_path
        check_path(raw_data_path)

        x_data_path = raw_data_path / params["x_data_dir"]
        params["x_data_path"] = x_data_path
        check_path(x_data_path)

        y_data_path = raw_data_path / params["y_data_dir"]
        params["y_data_path"] = y_data_path
        check_path(y_data_path)

        splits_path = raw_data_path / params["splits_dir"]
        params["splits_path"] = splits_path
        check_path(splits_path)

        # # ML data dir
        # ml_data_path = mainpath / params["ml_data_outdir"]
        # params["ml_data_path"] = ml_data_path
        # os.makedirs(ml_data_path, exist_ok=True)
        # check_path(ml_data_path)
        # os.makedirs(params["ml_data_outdir"], exist_ok=True)
        # check_path(params["ml_data_outdir"])

        # Models dir
        # os.makedirs(params["model_outdir"], exist_ok=True)
        # check_path(params["model_outdir"])

        # Infer dir
        # os.makedirs(params["infer_outdir"], exist_ok=True)
        # check_path(params["infer_outdir"])

        return params

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
        self.check_input_paths()
        self.check_output_dir()

        self.response_fname = "response.tsv"
        self.known_file_names = [self.response_fname]

        self.params = cfg.dict()
        self.sep = "\t"
        self.inp = self.params["y_data_files"]
        self.y_col_name = self.params["y_col_name"]
        self.canc_col_name = self.params["canc_col_name"]
        self.drug_col_name = self.params["drug_col_name"]

        # self.y_data_path = params["y_data_path"]/params["y_data_files"][0][0]
        # self.y_data_path = self.params["y_data_path"]
        # self.split_fpath = self.splits_path/split_file
        self.dfs = {}
        self.verbose = verbose
        if self.verbose:
            print(f"y_data_files: {params['y_data_files']}")
            print(f"y_col_name: {params['y_col_name']}")

        self.inp_fnames = []

    def load_data(self, verbose=False):
        """Load data from input directory."""
        self.verbose = verbose

        params = self.params
        params['x_data_path'] = self.x_data_path
        params['y_data_path'] = self.y_data_path
        params['splits_path'] = self.splits_path

        self.logger.debug("Loading Omics Data.")
        self.omics = drp.OmicsLoader(params)
        self.logger.info(self.omics)

        self.logger.debug("Loading Drug Data.")
        self.drugs = drp.DrugsLoader(params)
        self.logger.info(self.drugs)

        self.logger.debug("Loading Response Data.")
        self.train = drp.DrugResponseLoader(params,
                                            split_file=params["train_split_file"],
                                            verbose=False).dfs["response.tsv"]
        self.validate = drp.DrugResponseLoader(params,
                                               split_file=params["val_split_file"],
                                               verbose=False).dfs["response.tsv"]
        if params["test_split_file"] and os.path.exists(Path(self.splits_path) / params["test_split_file"]):
            self.test = drp.DrugResponseLoader(params,
                                               split_file=params["test_split_file"],
                                               verbose=False).dfs["response.tsv"]
        else:
            self.logger.warning(f"Test split file {
                                params['test_split_file']} does not exist.")

    def set_input_dir(self, input_dir: str) -> None:
        """Set input directory for Drug Response Prediction Benchmark."""

        # check if input_dir is Path object otherwise convert to Path object
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)

        self.input_dir = input_dir
        self.x_data_path = Path(input_dir) / "x_data"
        self.y_data_path = Path(input_dir) / "y_data"
        self.splits_path = Path(input_dir) / "splits"

    def set_output_dir(self, output_dir: str) -> None:
        """Set output directory for Drug Response Prediction Benchmark."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        self.output_dir = output_dir

    # Check all paths and directories are valid and exist, otherwise create them
    def check_input_paths(self) -> None:
        """Check input directory for Drug Response Prediction Benchmark. Return error if path does not exist.
        """
        if not Path(self.x_data_path).exists():
            raise FileNotFoundError(f"Path {self.x_data_path} does not exist.")
        if not Path(self.y_data_path).exists():
            raise FileNotFoundError(f"Path {self.y_data_path} does not exist.")
        if not Path(self.splits_path).exists():
            raise FileNotFoundError(f"Path {self.splits_path} does not exist.")

    def mkdir_input_dirs(self) -> None:
        self.x_data_path.mkdir(parents=True, exist_ok=True)
        self.y_data_path.mkdir(parents=True, exist_ok=True)
        self.splits_path.mkdir(parents=True, exist_ok=True)

    def check_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Get Benchmark Data from ftp site or URL and save to input directory. Retrieve recursively if necessary
    def get_benchmark_data(self) -> None:
        """Get Drug Response Prediction Benchmark data from ftp site or URL and save to input directory."""
        pass

    # Load Omics Data from input directory using drp module
    def load_omics_data(self, cfg) -> None:
        """Load omics data from input directory using drp module."""

        # Check if cfg is dict or BaseConfig object
        if isinstance(cfg, dict):
            raise TypeError("cfg Config object.")
        elif not isinstance(cfg, Config):
            raise TypeError("cfg must be a dict or Config object.")

        return drp.OmicsLoader(cfg)
        pass


class SingleDRPDataFrame(StringEnum):
    """
    Enum for specifying single drug response prediction dataframes.
    """

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
    """
    Enum for specifying datasets in single drug response prediction benchmarks.
    """
    CCLE = 'CCLE'
    CTRP = 'CTRPv2'
    GDSCv1 = 'GDSCv1'
    GDSCv2 = 'GDSCv2'
    gCSI = 'gCSI'


class SingleDRPMetric(StringEnum):
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
        self.super().__init__(
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
    CANCER_COL_NAME = "improve_sample_id"
    DRUG_COL_NAME = "improve_chem_id"

    def __init__(self):
        """
        Initializes the SingleDRPBenchmark instance with default values.
        """
        self._initialize()
        self._SPLITS_NUM = 10
        self._splits_ids = list(range(self._SPLITS_NUM))
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
            fname = '_'.join((str(self._dataset), 'all'))
            return f'{fname}.txt'
        filename = '_'.join((str(self._dataset), 'split',
                             str(self._split_id), str(self._stage)))
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
        dataframe_file = self._dataset2file_map[dataframe]
        loader_params = {}
        loader_params["x_data_canc_files"] = str([[dataframe_file]])
        loader_params["canc_col_name"] = self.CANCER_COL_NAME
        loader_params["x_data_path"] = os.path.join(
            self._benchmark_dir, 'x_data')
        omics_loader = drp.OmicsLoader(loader_params)
        return omics_loader.dfs[dataframe_file]

    def _load_drug_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Loads a drug-related dataframe based on the specified dataframe type.

        Args:
            dataframe (SingleDRPDataFrame): The type of drug-related dataframe to load.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        dataframe_file = self._dataset2file_map[dataframe]
        loader_params = {}
        loader_params["x_data_drug_files"] = str([[dataframe_file]])
        loader_params["drug_col_name"] = self.DRUG_COL_NAME
        loader_params["x_data_path"] = os.path.join(
            self._benchmark_dir, 'x_data')
        drug_loader = drp.DrugsLoader(loader_params)
        return drug_loader.dfs[dataframe_file]

    def _load_response_dataframe(self, dataframe: SingleDRPDataFrame):
        """
        Loads a response dataframe based on the specified dataframe type.

        Args:
            dataframe (SingleDRPDataFrame): The type of response dataframe to load.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        dataframe_file = self._dataset2file_map[dataframe]
        loader_params = {}
        loader_params["y_data_files"] = str([dataframe_file])
        loader_params["canc_col_name"] = self.CANCER_COL_NAME
        loader_params["drug_col_name"] = self.DRUG_COL_NAME
        loader_params["y_col_name"] = str(self._metric)
        loader_params["y_data_path"] = os.path.join(
            self._benchmark_dir, 'y_data')
        loader_params["splits_path"] = os.path.join(
            self._benchmark_dir, self._splits_dir)
        split_file = self._construct_splits_file_name()
        response_loader = drp.DrugResponseLoader(
            loader_params, split_file=split_file)
        df = response_loader.dfs[dataframe_file]
        cols_to_drop = [col for col in df.columns if col not in [
            self.CANCER_COL_NAME, self.DRUG_COL_NAME, str(self._metric)]]
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
        # np.unique(response_df[key_col_name])
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

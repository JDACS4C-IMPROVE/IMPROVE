from improve.Benchmarks import Base as Base

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

import pandas as pd
import improve.drug_resp_pred as drp

class DRP(Base.Base):
    """Class to handle configuration files for Drug Response Prediction."""
    # Set section for config file
    section = 'DRPBenchmark_v1.0'

    # Default format for logging
    FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
    logging.basicConfig(format=FORMAT)
    logger=logging.getLogger('DRP')
    # logger=logging.getLogger(__name__)
    logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.ERROR))

    # Set options for command line
    drp_options = [
        {
            'name':'benchmark_dir',
            'default' : './' ,
            'type': str,
            'help':'Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.'
        },
        {
            'name':'training_index_file',
            'default' : 'training.idx',
            'type': str,
            'help':'index file for training set [numpy array]'
        },
        {
            'name':'validation_index_file',
            'default' : 'validation.idx' ,
            'type': str,
            'help':'index file for validation set [numpy array]',
        },
        {
            'name':'testing_index_file',
            'default' : 'testing.idx' ,
            'type': str,
            'help':'index file for testiing set [numpy array]',
        },
        {
            'name':'data',
            'default' : 'data.parquet' ,
            'type': str,
            'help':'data file',
        },
        {
            'name':'input_type',
            'default' : 'BenchmarkV1' ,
            'choices' : ['parquet', 'csv', 'hdf5', 'npy', 'BenchmarkV1'],
            'metavar' : 'TYPE',
            'help' : 'Sets the input type. Default is BenchmarkV1. Other options are parquet, csv, hdf5, npy'
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

    def init(self, cfg , verbose=False):
        """Initialize Drug Response Prediction Benchmark. Takes Config object as input."""
        
        if cfg.log_level:
            self.logger.setLevel(cfg.log_level)
            self.logger.debug(f"Log level set to {cfg.log_level}.")
        else:
             self.logger.warning("No log level set in Config object. Using default log level.")

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
            self.logger.warning(f"Test split file {params['test_split_file']} does not exist.")



        
        
        




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
    def load_omics_data(self,cfg) -> None:
        """Load omics data from input directory using drp module."""

        # Check if cfg is dict or BaseConfig object
        if isinstance(cfg, dict):
            raise TypeError("cfg Config object.")
        elif not isinstance(cfg, Config):
            raise TypeError("cfg must be a dict or Config object.")

        return drp.OmicsLoader(cfg)
        pass
        
if __name__ == "__main__":
    drp = DRP()
    drp.set_input_dir("input_dir")
    print(drp.__dict__)
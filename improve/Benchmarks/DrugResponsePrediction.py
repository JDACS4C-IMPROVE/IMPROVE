from improve.Benchmarks import Base as Base

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

class DRP(Base.Base):
    """Class to handle configuration files for Drug Response Prediction."""
    # Set section for config file
    section = 'DRPBenchmark_v1.0'

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
        self.options = DRP.drp_options
        self.input_dir = None
        self.x_data_path = None
        self.y_data_path = None
        self.splits_path = None

    def set_input_dir(self, input_dir: str) -> None:
        """Set input directory for Drug Response Prediction Benchmark."""

        # check if input_dir is Path object otherwise convert to Path object
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)

        self.input_dir = input_dir
        self.x_data_path = Path(input_dir) / "x_data_dir"
        self.y_data_path = Path(input_dir) / "y_data_dir"
        self.splits_path = Path(input_dir) / "splits_dir"

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
        
        
if __name__ == "__main__":
    drp = DRP()
    drp.set_input_dir("input_dir")
    print(drp.__dict__)
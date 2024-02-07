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
        self.cli.parser.add_argument('--supplemental', nargs=2, action='append', metavar=('TYPE', 'FILE'), 
                              type=str, help='Supplemental data FILE and TYPE. FILE is in INPUT_DIR.')
        benchmark_cli=self.cli.parser.add_argument_group('DRPBenchmark_v1.0', 'Options for drug response prediction benchmark v1.0')
        benchmark_cli.add_argument('--benchmark', action='store_true', help='Use DRPBenchmark_v1.0')
        benchmark_cli.add_argument('--benchmark_dir', metavar='DIR', type=str, dest="benchmark_dir",
                                  default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
                                  help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
if __name__ == "__main__":
    drp = DRP()
    print(drp.params)
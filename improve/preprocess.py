import os
import sys
from improve.config import Config

class Params:
    pass



class Preprocess(Config):
    """Class to handle configuration files for Preprocessing."""

    # Set section for config file
    section = 'Preprocess'

    # Set options for command line
    preprocess_options = [
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
        self.options = Preprocess.preprocess_options
        self.cli.parser.add_argument('--supplemental', nargs=2, action='append', metavar=('TYPE', 'FILE'), 
                              type=str, help='Supplemental data FILE and TYPE. FILE is in INPUT_DIR.')
        benchmark_cli=self.cli.parser.add_argument_group('DRPBenchmark_v1.0', 'Options for drug response prediction benchmark v1.0')
        benchmark_cli.add_argument('--benchmark', action='store_true', help='Use DRPBenchmark_v1.0')
        benchmark_cli.add_argument('--benchmark_dir', metavar='DIR', type=str, dest="benchmark_dir",
                                    default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
                                    help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
        subparsers=self.cli.parser.add_subparsers(dest='subparser_name')
        benchmark=subparsers.add_parser('benchmark', help='Use DRPBenchmark_v1.0')
        benchmark.add_argument('--benchmark', action='store_true', help='Use DRPBenchmark_v1.0')

        default=subparsers.add_parser('file', help='Use generic file import')
        default.add_argument('--measurments', type=str,  help='File with measurements')
        default.add_argument('--supplemental', nargs=2, action='append', metavar=('TYPE', 'FILE'), 
                              type=str, help='Supplemental data FILE and TYPE. FILE is in INPUT_DIR.')
        default.add_argument('--features', type=str,  help='File name for features. Default is features.parquet')
        default.add_argument('--input_type', type=str,  default="CSV", help='Sets the input type. Default is CSV. Other options are parquet, csv, hdf5, npy')
        default.add_argument('--output_type', type=str,  default="parquet", help="Sets the output type. Default is parquet. Other options are parquet, csv, hdf5, npy")


    def load_data(self, file):
        """Load data from a file."""
        pass

    def load_measurements(self, file):
        """Load measurements from a file."""
        pass

    def load_supplemental(self, file=None, type=None):
        """Load supplemental data from a file."""
        pass

    def save_features(self, file , type="Parquet"):
        """Save features to a file."""
        pass

    # def set_param(self, key, value):
    #     """Set a parameter in the Preprocessing config."""
    #     return super().set_param(Preprocess.section, key, value)
    
    def get_param(self, key):
        """Get a parameter from the Preprocessing config."""
        return super().set_param(Preprocess.section, key)


    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        """Initialize Command line Interfcace and config for Preprocessing."""
        if additional_definitions :
            self.options = self.options + additional_definitions

        return super().initialize_parameters(pathToModelDir, section, default_config, default_model, self.options , required)


    

if __name__ == "__main__":
    p=Preprocess()
    p.initialize_parameters(pathToModelDir=".")

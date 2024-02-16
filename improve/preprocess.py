import os
import sys
import logging
from improve.config import Config

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)


class Params:
    pass



class Preprocess(Config):
    """Class to handle configuration files for Preprocessing."""

    # Set section for config file
    section = 'Preprocess'

    # Set options for command line
    preprocess_options = [
        # {
        #     'name':'training_index_file',
        #     'default' : 'training.idx',
        #     'type': str,
        #     'help':'index file for training set [numpy array]'
        # },
        # {
        #     'name':'validation_index_file',
        #     'default' : 'validation.idx' ,
        #     'type': str,
        #     'help':'index file for validation set [numpy array]',
        # },
        # {
        #     'name':'testing_index_file',
        #     'default' : 'testing.idx' ,
        #     'type': str,
        #     'help':'index file for testiing set [numpy array]',
        # },
        # {
        #     'name':'data',
        #     'default' : 'data.parquet' ,
        #     'type': str,
        #     'help':'data file',
        # },
        # {
        #     'name':'input_type',
        #     'default' : 'BenchmarkV1' ,
        #     'choices' : ['parquet', 'csv', 'hdf5', 'npy', 'BenchmarkV1'],
        #     'metavar' : 'TYPE',
        #     'help' : 'Sets the input type. Default is BenchmarkV1. Other options are parquet, csv, hdf5, npy'
        # }
    ]

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("Preprocess")
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.INFO))
      
       
        self.options = Preprocess.preprocess_options
        # self.cli.parser.add_argument('--supplemental', nargs=2, action='append', metavar=('TYPE', 'FILE'), 
        #                       type=str, help='Supplemental data FILE and TYPE. FILE is in INPUT_DIR.')
        # benchmark_cli=self.cli.parser.add_argument_group('DRPBenchmark_v1.0', 'Options for drug response prediction benchmark v1.0')
        # benchmark_cli.add_argument('--benchmark', action='store_true', help='Use DRPBenchmark_v1.0')
        # benchmark_cli.add_argument('--benchmark_dir', metavar='DIR', type=str, dest="benchmark_dir",
        #                             default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
        #                             help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
        # Set subparser for benchmark and file
        subparsers=self.cli.parser.add_subparsers(dest='subparser_name')
        
        # Benchmark subparser
        benchmark=subparsers.add_parser('benchmark', help='Use DRPBenchmark_v1.0')
        # benchmark.add_argument('--benchmark', action='store_true', help='Use DRPBenchmark_v1.0')
        benchmark.add_argument('--benchmark_type', choices=['DRP', 'Default'], help='Specify benchmark format, e.g. DRP for DRPBenchmark_v1.0')
        benchmark.add_argument('--benchmark_dir', metavar='DIR', type=str, dest="benchmark_dir",
                                    default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
                                    help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
        drp=benchmark.add_argument_group('DRPBenchmark_v1.0', 'Options for drug response prediction benchmark v1.0')
        drp.add_argument('--drp', action='store_true', help='Use DRPBenchmark_v1.0')
        drp.add_argument('--drp_dir', metavar='DIR', type=str, dest="benchmark_dir",
                                    default=os.getenv("IMPROVE_BENCHMARK_DIR" , "./"), 
                                    help='Base directory for DRPBenchmark_v1.0 data. Default is IMPROVE_BENCHMARK_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        
        # drp.add_argument("--raw_data_dir", type=str, dest="input_dir" , help="Data dir name that stores the raw data, same as input_dir. The directory includes x data, y data, and splits.")
        drp.add_argument("--x_data_dir", type=str, default="x_data" , help="Dir name that contains the files with features data (x data). Default is ${input_dir}/x_data.")
        drp.add_argument("--y_data_dir", type=str, default="y_data" , help="Dir name that contains the files with target data (y data). Default is ${input_dir}/y_data.")
        drp.add_argument("--splits_dir", type=str, default="splits" , help="Dir name that contains files that store split ids of the y data file.")
        # drp.add_argument("--x_data_file", type=str, default="x.parquet" , help="File name for features. Default is x.parquet.")
        # drp.add_argument("--y_data_file", type=str, default="y.parquet" , help="File name for target. Default is y.parquet.")
        drp.add_argument("--pred_col_name_suffix", 
                        type=str, default="_pred" , 
                        help="Suffix to add to a column name in the y data file to identify predictions made by the model (e.g., if y_col_name is 'auc', then a new column that stores model predictions will be added to the y data file and will be called 'auc_pred').")
        drp.add_argument("--train_split_file", default="train_split.txt" , 
                         help="File name for the file that contains training split ids. Default is train_split.txt.")
        drp.add_argument("--val_split_file", default="val_split.txt" , 
                         help="File name for the file that contains validation split ids. Default is val_split.txt.")
        drp.add_argument("--test_split_file", default="test_split.txt" , 
                         help="File name for the file that contains test split ids. Default is test_split.txt.")    
        # drp.add_argument("--ml_data_outdir" , dest="output_dir", type=str, 
                        #  help="Output dir name for the preprocessed data. Same as output_dir. Default is ${input_dir}.")
        drp.add_argument("--feature_data_format", type=str, default="parquet" ,
                        help="Output format for the preprocessed data. Default is parquet.")
        drp.add_argument("--y_col_name", type=str, default="auc" , 
                         help="Column name in the y data file (e.g., response.tsv), that represents the target variable that the model predicts. In drug response prediction problem it can be IC50, AUC, and others.")
        drp.add_argument("--y_data_suffix" , default="y_data" , 
                         help="Suffix to compose file name for storing true y dataframe. Default is y_data.")
        drp.add_argument("--y_data_files", nargs='+', type=str, default=['response.tsv'] ,
                         help="List of files that contain the y (prediction variable) data. Example: ['response.tsv']")
        # x_data_canc_files , input is a list of tuples, each tuple has two elements, the first element is the file name and the second element is a comma separated list of gene system identifiers.
        drp.add_argument("--x_data_canc_file", nargs=2, action="append", metavar=("FILE", "IDENTIFIERS"), 
                         default=[],
                         help="List of tuples, each tuple has two elements, the first element is the file name and the second element is a comma separated list of gene system identifiers. Example: 'cancer_copy_number.tsv', 'Ensembl,Entrez'") 
        drp.add_argument("--canc_col_name", type=str, default="improve_sample_id" , 
                         help="Column name in the y data file (reponse) that represents the cancer sample identifier. Default is improve_sample_id.")
       
        drp.add_argument("--drug_col_name", type=str, default="improve_chem_id" , help="Column name in the y data file (reponse) that represents the drug identifier. Default is improve_chem_id.")


        # File subparser
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
        return super().get_param(Preprocess.section, key)

    def dict(self):
        """Get the Preprocessing config as a dictionary."""
        return super().dict(Preprocess.section)

    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        """Initialize Command line Interfcace and config for Preprocessing."""
        self.logger.debug("Initializing parameters for Preprocessing.")
        if additional_definitions :
            self.options = self.options + additional_definitions

        p = super().initialize_parameters(pathToModelDir, section, default_config, default_model, self.options , required)
        print(self.get_param("log_level"))
        self.logger.setLevel(self.get_param("log_level"))
        return p


    

if __name__ == "__main__":
    p=Preprocess()
    p.initialize_parameters(pathToModelDir=".")
    print(p.dict())

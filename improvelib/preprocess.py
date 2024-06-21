import os
import sys
import logging
from improvelib.config import Config
from improvelib.Benchmarks.DrugResponsePrediction import DRP as BenchmanrkDRP


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
      
        # check usage of self.options
        self.options = Preprocess.preprocess_options


        self.cli.parser.add_argument('--supplemental', nargs=2, action='append', metavar=('TYPE', 'FILE'), 
                              type=str, help='Supplemental data FILE and TYPE. FILE is in INPUT_DIR.')
       
        # Set subparser for benchmark and file
        # subparsers=self.cli.parser.add_subparsers(dest='subparser_name')
        
        # Add options for Benchmark Data Format 
        p = self.cli.parser
        benchmark = p.add_argument_group('Benchmark Data Format', 'Options for benchmark data format')       

        #     
        benchmark.add_argument("--x_data_dir", type=str, default="x_data" , help="Dir name that contains the files with features data (x data). Default is ${input_dir}/x_data.")
        benchmark.add_argument("--y_data_dir", type=str, default="y_data" , help="Dir name that contains the files with target data (y data). Default is ${input_dir}/y_data.")
        benchmark.add_argument("--splits_dir", type=str, default="splits" , help="Dir name that contains files that store split ids of the y data file.")
        benchmark.add_argument("--supplement_dir", type=str, default=None , help="Dir name that contains supplemental data.")

        drp=benchmark.add_argument_group('Drug Response Prediction', 'Options for drug response prediction benchmark data')
       
        # drp.add_argument("--raw_data_dir", type=str, dest="input_dir" , help="Data dir name that stores the raw data, same as input_dir. The directory includes x data, y data, and splits.")
  
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
    
    def set_params(self, key=None, value=None):
        print( "set_params" + type(self))
        return super().set_param(Preprocess.section, key, value)
    
    def set_param(self, key=None, value=None):
        return super().set_param(Preprocess.section, key, value)

    def dict(self):
        """Get the Preprocessing config as a dictionary."""
        return super().dict(Preprocess.section)

    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        """Initialize Command line Interfcace and config for Preprocessing."""
        self.logger.debug("Initializing parameters for Preprocessing.")
        print( "initialize_parameters" + str(type(self)) )

        if additional_definitions :
            self.options = self.options + additional_definitions

       
        p = super().initialize_parameters(pathToModelDir, section, default_config, default_model, self.options , required)
        print(self.get_param("log_level"))
        self.logger.setLevel(self.get_param("log_level"))
        return p

    def load_data(self, loader=None):
        """Get data from files or benchmarks."""

        self.logger.debug("Loading data from preprocess.")

        # Defined here but set after loading data. Value is an object with a dfs attribute.
        self.omics = None
        self.drugs = None

        try:
            if self.get_param("subparser_name") is None or self.get_param("subparser_name") == "":
                logger.error("Subparser name is not set.")
                # throw error
                raise ValueError("Missing mandatory positional parameter: subparser_name.")
            else:
                logger.info(f"Subparser name: {self.get_param('subparser_name')}")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            sys.exit(1)
        
        # if subparser_name is benchmark, then use the BenchmarkDRP class
        # to load the data
        if self.get_param("subparser_name") == "benchmark":
            DRP = BenchmanrkDRP()
            self.drp = DRP
            DRP.init(self)
            DRP.set_input_dir(self.get_param("input_dir"))
            DRP.set_output_dir(self.get_param("output_dir"))
            # Check all paths and directories are valid and exist
            DRP.check_input_paths()
            # Create output dir for model input data (to save preprocessed ML data)
            DRP.check_output_dir()
            # Load data
            logger.debug("Loading from DRP class")
            DRP.load_data(verbose=True)
            self.set_param("x_data_path", DRP.x_data_path)
            self.set_param("y_data_path", DRP.y_data_path)  
            self.set_param("splits_path", DRP.splits_path)
            # print(DRP.__dict__)
        else:
            raise ValueError("Not implemented.")

        self.omics = None
        self.drugs = None

        self.omics = self.drp.omics
        self.drugs = self.drp.drugs


        pass
    

if __name__ == "__main__":
    p=Preprocess()
    p.initialize_parameters(pathToModelDir=".")
    print(p.dict())

# Library for drug response prediction specific functions for preprocessing


# Create new class for DrugResponsePrediction inheriting from Preprocess

from improvelib.preprocess import Preprocess
import logging
import os


FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

class DrugResponsePrediction(Preprocess):

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger("Preprocess.DRP")
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.INFO))
        self.logger.debug("Initializing DrugResponsePrediction")

       
        # Set subparser for benchmark and file
        # subparsers=self.cli.parser.add_subparsers(dest='subparser_name')
        
        # Add options for Drug Response Prediction
        p = self.cli.parser
        drp=p.add_argument_group('Drug Response Prediction', 'Options for drug response prediction benchmark data')
       
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



    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        """Initialize Command line Interfcace and config for /DRP."""
        self.logger.debug("Initializing parameters for Preprocessing.DRP.")
        print( "initialize_parameters" + str(type(self)) )

        if additional_definitions :
            self.options = self.options + additional_definitions

       
        p = super().initialize_parameters(pathToModelDir, section, default_config, default_model, self.options , required)
        print(self.get_param("log_level"))
        self.logger.setLevel(self.get_param("log_level"))
        return p




if __name__ == "__main__":
    drp = DrugResponsePrediction()

    # Model specific parameters
    model_params = [ { 'name' : "model_name"} , { 'name' : 'model_version'} ]

    # Initialize parameters for preprocessing (application + model specific)
    drp.initialize_parameters( 
        pathToModelDir=None, 
        default_config=None,
        additional_definitions=model_params,
        required=["model_name", "model_version"] )
      
    print(drp.dict())

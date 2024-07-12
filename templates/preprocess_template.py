import sys
from pathlib import Path
from typing import Dict

# from improvelib import class for cli and config
# General parent class for preprocessing
from improvelib.preprocess.base import Preprocess
from improvelib.preprocess.drug_response_prediction import Preprocess as DrugResponsePredictionPreprocess

import pandas as pd



# Global variables
filepath = Path(__file__).resolve().parent # [Req]

my_params_example    = [                                                    # see argparse.ArgumentParser.add_argument() for more information
    {
        "name": "y_data_files",                                                 # name of the argument
        "type": str,                                                            # type of the argument
        "nargs": "+",                                                           # number of arguments that should be consumed
        "help": "List of files that contain the y (prediction variable) data.", # help message
    },
    {
        "name": "supplement",                                               # name of the argument
        "type": str,                                                        # type of the argument
        "nargs": 2,                                                         # number of arguments that should be consumed
        "metavar": ("FILE", "TYPE"),                                        # name of the argument in usage messages
        "action": "append",                                                 # action to be taken when this argument is encountered
        "help": "Supplemental data tuple FILE and TYPE. FILE is in INPUT_DIR.",   # help message
    }
]

# Default functions for all improve scripts are main and run
# main initializes parameters and calls run with the parameters
def run(params: Dict, cfg: Preprocess = None):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
        cfg (Preprocess): Preprocess object. Default is None. 

    Returns:
        str: status of the preprocessing.
    """

    # Check if params is dict or of type Preprocess
    if not isinstance(params, dict):
        cfg = params
        params = cfg.dict()


    cfg.logger.info("Running preprocessing.") if cfg else print("Running preprocessing.")
    # Load data from input directory, the input directory is defined in the configuration file
    # and is accessible through the cfg object
    # The input directory is the directory where the raw data is stored in the IMPROVE Benchmark Format
    # The raw data is stored in the input directory in the following format:
    # x_data: directory that contains the files with features data (x data)
    # y_data: directory that contains the files with target data (y data)
    # splits: directory that contains files that store split ids of the y data file
    # supplement: directory that contains supplemental data (optional) and not used in this example or part of the IMPROVE Benchmark Format
    cfg.logger.debug(f"Loading data from {cfg.input_dir}.") if cfg else print(f"Loading data from {params['input_dir']}.")

    ###### Place your code here ######
    
    # Load x data, e.g.
    # data = cfg.loader.load_data()    # Load y data, e.g. 

    # y_data = pd.read_csv(Path(cfg.input_dir,"y_data","response.tsv"), sep="\t")
    # y_data = pd.read_csv(params['input_dir'] / "y_data" / "response.tsv", sep="\t")
    # y_data = cfg.loader.load_data()


    # Transform data
   

    # Save data
    # y_data.to_csv(params["output_dir"] / "y_data.csv", index=False)


    return params["output_dir"]




# Default functions for all improve scripts are main and run
# main initializes parameters and calls run with the parameters
# run is the main function that executes the script
def main(args):
    
    # List of custom parameters, if any
    # can be list or file in json or yaml format, e.g. :
    # my_additional_definitions = [{"name": "param_name", "type": str, "default": "default_value", "help": "help message"}]
    # my_additional_definitions = None
    # my_additional_definitions = Path("custom_params.json")
    # my_additional_definitions = filepath/"custom_params.json"

    # Set additional_definitions to None if no custom parameters are needed
   
    # Example of file with custom parameters, the file should be in json or yaml format and is located in the same directory as the script
    # additional_definitions = Path("custom_params.json")
    my_params_file_example = filepath/"common_params.yml"

    # Exampple 1: Initialize parameters using the Preprocess class

    # Initialize parameters using the Preprocess class
    cfg = Preprocess()
    params = cfg.initialize_parameters(filepath, 
                                       default_config="default.cfg",
                                       additional_definitions=my_params_example,
                                       required=None
                                       )
    
    # run task, passing params to run for backward compatibility, cfg could be used instead and contains the same information as params
    # in addition to the parameters, the cfg object provides access to the logger, the config object and data loaders
    # default is run(params)
    run(cfg)


    # Example 2: Initialize parameters using the DrugResponsePrediction class

    # Initialize parameters using the DrugResponsePrediction class
    # the DrugResponsePrediction class is a subclass of Preprocess
    cfg = DrugResponsePredictionPreprocess()
    params = cfg.initialize_parameters(filepath, 
                                       default_config="default.cfg",
                                       additional_definitions=my_params_file_example,
                                       required=None
                                       )
  
    status = run(params)
    cfg.logger.info(f"Preprocessing completed. Data saved in {cfg.output_dir}")
    return status

if __name__ == "__main__":
    main(sys.argv[1:])

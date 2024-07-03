import os
import sys
from pathlib import Path
from typing import Dict

# IMPROVE imports - use the appropriate import for your application
# from improvelib.applications.drug_response_prediction import PreprocessConfig

from improvelib.initializer.stage_config import PreprocessConfig
# from improvelib.Preprocess import Config as PreprocessConfig

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Required functions for all improve scripts are main() and run().
# main() initializes parameters and calls run() with the parameters.


def run(cfg=None, logger=None):
    """ Run data preprocessing.

    Args:
        cfg: Stage specific config object. Provides logger and a dict of parameters for the preprocessing parsed from cli and config file.

    Returns:
        str: status of the preprocessing.
    """
    params = cfg.params
    logger = cfg.logger if logger is None else logger

    logger.info("Running preprocessing.") if logger else print(
        "Running preprocessing.")
    logger.debug(f"Loading data from {params['input_dir']}.")

    # see template 
    return "success"

def example_parameter_from_file():
    # Example: Initialize parameters using a configuration file
    # In this example the custom parameters are defined in a file named custom_params.json 
    # in the same directory as the script.

    cli_config_file = filepath/"model_params.json"

    return cli_config_file


def example_parameter_from_dictionary():
    # Example: Initialize parameters using custom list

    # Initialize parameters using custom parameters list following the options 
    # defined in Python argparse.ArgumentParser format
    my_params_example = [
        # see argparse.ArgumentParser.add_argument() for more information
        {
            # name of the argument
            "name": "y_data_files",
            # type of the argument
            "type": str,
            # number of arguments that should be consumed
            "nargs": "+",
            # help message
            "help": "List of files that contain the y (prediction variable) data.",
        },
        {
            # name of the argument
            "name": "supplement",
            # type of the argument
            "type": str,
            # number of arguments that should be consumed
            "nargs": 2,
            # name of the argument in usage messages
            "metavar": ("FILE", "TYPE"),
            # action to be taken when this argument is encountered
            "action": "append",
            "help": "Supplemental data tuple FILE and TYPE. FILE is in INPUT_DIR.",   # help message
        }
    ]
    
    return my_params_example





# Required functions for all improve model scripts are main() and run().
# main() initializes parameters and calls run() with the parameters.
# run() is the function that executes the primary processing of the model script and can be imported into other scripts.
def main(args):

    # Model specific parameters from LGBM for preprocessing for example
    # The example below shows how to initialize parameters using a configuration file or a dictionary.

    model_cli_config_params = example_parameter_from_file()
    model_cli_config_params = example_parameter_from_dictionary()
 

    cfg = PreprocessConfig()
    params = cfg.initialize_parameters('Your model name', # e.g. 'LGBM
                                       default_config=None, # e.g. default.cfg or default.ini
                                       additional_cli_section='Your model name', # e.g. 'LGBM
                                       additional_definitions=model_cli_config_params,
                                       required=None
                                       )

    status = run(params_lgbm, logger_lgbm)
    
    cfg.logger.info(f"Preprocessing completed with {status}. Data saved in {params['output_dir']}") if cfg.logger else print(
        f"Preprocessing completed. Data saved in {params['output_dir']}")
    


if __name__ == "__main__":
    main(sys.argv[1:])

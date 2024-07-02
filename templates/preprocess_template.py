import sys
from pathlib import Path
from typing import Dict

from improvelib.initializer.stage_config import PreprocessConfig
# from improvelib.preprocess import Config as Preprocess

import pandas as pd


# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Define parameters for the script
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
            # default value of the argument
            "default": [ ('a' , 'b')],
            "help": "Supplemental data tuple FILE and TYPE. FILE is in INPUT_DIR.",   # help message
        }
    ]


# Default functions for all improve scripts are main and run
# main initializes parameters and calls run with the parameters
# run is the main function that executes the script or can be imported and used in other scripts

def run(params: Dict, logger=None):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
        cfg (Preprocess): Preprocess object. Default is None. 

    Returns:
        str: status of the preprocessing.
    """

    logger.info("Running preprocessing.") if logger else print(
        "Running preprocessing.")

     ###### Place your code here ########


    logger.debug(f"Loading data from {params['input_dir']}.")
    ###### Place your code here ######

    logger.debug(f"Creating data from {params['input_dir']}.")
    ###### Place your code here ######
    
    logger.info(f"Run completed. Data saved in {params['output_dir']}")
    return "success"







def main(args):
    """ Main function for preprocessing."""
    
    cfg = PreprocessConfig()
    params = cfg.initialize_parameters(filepath,
                                    default_config="default.cfg",
                                    # additional_cli_section='My section',
                                    additional_definitions=my_params_example,
                                    required=None
                                    )


    status2 = run(params, cfg.logger)


if __name__ == "__main__":
    main(sys.argv[1:])

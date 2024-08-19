import sys
from pathlib import Path
from typing import Dict

from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Define parameters for the script
test_model_preprocess_params = [
        # see argparse.ArgumentParser.add_argument() for more information
        {
            # name of the argument
            "name": "preprocess_test_var",
            # type of the argument
            "type": str,
            # help message
            "help": "Test variable for preprocess.",
            "default": "prep", # must include default, otherwise not defined
        },
        {   
            "name": "split",
            "type": list,
            "default": ['0'],
            "help": "Split number for preprocessing"
        },
        {
            "name": "only_cross_study",
            "type": bool,
            "default": False,
        },
        {
            "name": "study_number",
            "type": int,
            "default": 1,
        },
        {
            "name": "train_percent",
            "type": float,
            "default": 0.8,
        },
    ]


# Default functions for all improve scripts are main and run
# main initializes parameters and calls run with the parameters
# run is the main function that executes the script or can be imported and used in other scripts
'''
def run(params: Dict, logger=None):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
        cfg (Preprocess): Preprocess Config object. Default is None. 

    Returns:
        str: status of preprocessing ("Success.").
    """

    #logger.info("Running preprocessing.") if logger else print(
    #    "Running preprocessing.")
    ###### Place your code here ########

    #logger.debug(f"Loading data from {params['input_dir']}.")
    ###### Place your code here ######

    #logger.debug(f"Creating data from {params['input_dir']}.")
    ###### Place your code here ######
    
    #logger.info(f"Run completed. Data saved in {params['output_dir']}.")
    return "Success."
'''

def main(args):
    """ Main function for preprocessing."""
    
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(filepath,
                                    default_config="test_default_1.cfg",
                                    # additional_cli_section='My section',
                                    additional_definitions=test_model_preprocess_params,
                                    required=None
                                    )
    return params # instead of parsing output string


if __name__ == "__main__":
    main(sys.argv[1:])
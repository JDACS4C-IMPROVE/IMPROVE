import sys
from pathlib import Path
from typing import Dict

from improvelib.applications.drug_response_prediction.config import DRPTrainConfig

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Define parameters for the script
test_model_train_params = [
        # see argparse.ArgumentParser.add_argument() for more information
        {
            # name of the argument
            "name": "train_test_var",
            # type of the argument
            "type": int,
            # help message
            "help": "Test variable for train.",
            "default": 5,
        }
    ]


# Default functions for all improve scripts are main and run
# main initializes parameters and calls run with the parameters
# run is the main function that executes the script or can be imported and used in other scripts

def run(params: Dict, logger=None):
    """ Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
        cfg (Train): Train Config object. Default is None. 

    Returns:
        str: status of training ("Success.").
    """

    logger.info("Running model training.") if logger else print(
        "Running model training.")
    ###### Place your code here ########

    logger.debug(f"Loading data from {params['input_dir']}.")
    ###### Place your code here ######

    logger.debug(f"Training model.")
    ###### Place your code here ######
    
    logger.info(f"Run completed. Model and results saved in {params['output_dir']}.")
    return "Success."


def main(args):
    """ Main function for training."""
    
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(filepath,
                                    default_config="test_default_1.cfg",
                                    # additional_cli_section='My section',
                                    additional_definitions=test_model_train_params,
                                    required=None
                                    )
    return params


if __name__ == "__main__":
    main(sys.argv[1:])
import sys
from pathlib import Path
from typing import Dict

from improvelib.applications.drug_response_prediction.config import DRPInferConfig

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Define parameters for the script
test_model_infer_params = [
        # see argparse.ArgumentParser.add_argument() for more information
        {
            # name of the argument
            "name": "infer_test_var",
            # type of the argument
            "type": str,
            # help message
            "help": "Test variable for infer.",
            "default": "infer", # must include default, otherwise not defined
        }
    ]


# Default functions for all improve scripts are main and run
# main initializes parameters and calls run with the parameters
# run is the main function that executes the script or can be imported and used in other scripts

def run(params: Dict, logger=None):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.
        cfg (Infer): Infer Config object. Default is None. 

    Returns:
        str: status of inference ("Success.").
    """

    logger.info("Running model inference.") if logger else print(
        "Running model inference.")

    ###### Place your code here ########

    logger.debug(f"Loading model and data from {params['input_dir']}.")
    ###### Place your code here ######

    logger.debug(f"Inference results.")
    ###### Place your code here ######
    
    logger.info(f"Run completed. Results saved in {params['output_dir']}.")
    return "Success."


def main(args):
    """ Main function for inference."""
    
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(filepath,
                                    default_config="test_default_1.cfg",
                                    # additional_cli_section='My section',
                                    additional_definitions=test_model_infer_params,
                                    required=None
                                    )
    return params


if __name__ == "__main__":
    main(sys.argv[1:])
import sys
from pathlib import Path
from typing import Dict

from improvelib.applications.drug_response_prediction.config import DRPTrainConfig

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Define parameters for the script
test_model_train_params = [
        {
            "name": "train_test_var",
            "type": int,
            "help": "Test variable for train.",
            "default": 5,
        },
        {   
            "name": "split",
            "type": int,
            "nargs" : "+",
            "default": [0],
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
        {
            "name": "variable_name",
            "type": str,
            "default": "",
        },
    ]


def main(args):
    """ Main function for training."""
    
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(filepath,
                                    default_config="default.cfg",
                                    additional_definitions=test_model_train_params,
                                    required=None
                                    )
    return params

if __name__ == "__main__":
    main(sys.argv[1:])

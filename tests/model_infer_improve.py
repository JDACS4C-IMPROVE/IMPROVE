import sys
from pathlib import Path


from improvelib.applications.drug_response_prediction.config import DRPInferConfig

# Global variables
filepath = Path(__file__).resolve().parent  # [Req]

# Define parameters for the script
test_model_infer_params = [
        {
            "name": "infer_test_var",
            "type": str,
            "help": "Test variable for infer.",
            "default": "infer",
        },
                {   
            "name": "split",
            "nargs" : "+",
            "type": str,
            "default": ['0'],
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
        {
            "name": "json_scores_suffix",
            "type": str,
            "default": "",
        }
    ]

def main(args):
    """ Main function for inference."""
    
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(filepath,
                                    default_config="default.cfg",
                                    additional_definitions=test_model_infer_params,
                                    required=None
                                    )
    return params

if __name__ == "__main__":
    main(sys.argv[1:])
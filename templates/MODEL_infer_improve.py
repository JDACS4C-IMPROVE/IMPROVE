""" TEMPLATE for Inference for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["output_dir"].

1. Predictions on test data.
   Raw model predictions calculated using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from model_params_def import infer_params

# Model-specific imports, as needed

filepath = Path(__file__).resolve().parent # [Req]


# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for test set and build model path
    # --------------------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"], 
        model_file_format=params["model_file_format"], 
        model_dir=params["output_dir"])

    # --------------------------------------------------------------------
    # Load inference data (ML data)
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=your_test_true, 
        y_pred=your_test_predicted, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=your_test_true, 
            y_pred=your_test_predicted, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="MODEL_params.txt",
        additional_definitions=infer_params,
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

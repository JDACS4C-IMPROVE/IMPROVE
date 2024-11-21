""" Train GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this train script are saved in params["output_dir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].
   For GraphDRP, the saved model:
        model.pt

2. Predictions on val data. 
   Raw model predictions calcualted using the trained model on val data. The
   predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in val_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics
from model_params_def import train_params

# Model-specific imports, as needed

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"], 
        model_file_format=params["model_file_format"], 
        model_dir=params["output_dir"])

    # --------------------------------------------------------------------
    # Load model input data (ML data) for train and val
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Prepare model
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Train. Iterate over epochs.
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # --------------------------------------------------------------------
    frm.store_predictions_df(
        y_true=your_val_true, 
        y_pred=your_val_predicted, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # --------------------------------------------------------------------
    # [Req] Compute performance scores
    # --------------------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=your_val_true, 
        y_pred=your_val_predicted, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="MODEL_params.txt",
        additional_definitions=train_params)
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

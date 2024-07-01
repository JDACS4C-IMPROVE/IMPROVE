from model_parameters import model_params
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from improvelib.initializer.stage_config import InferConfig

# [Req] IMPROVE imports
from improvelib.tools import utils

filepath = Path(__file__).resolve().parent  # [Req]
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]

# ---------------------
# [Req] Parameter lists
# ---------------------
# model_infer_params
#
# The values for the parameters in a list should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().


# Model-specific params (Model: LightGBM)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# ---------------------

# [Req]


def run(params: Dict):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    utils.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = utils.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    te_data = pd.read_parquet(Path(params["test_ml_data_dir"])/test_data_fname)

    # Test data
    yte = te_data[[params["y_col_name"]]]

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = utils.build_model_path(
        params, model_dir=params["model_dir"])  # [Req]

    ########################################################
    ############# Your model's code is here ################
    ########################################################
    # Load LightGBM
    model = lgb.Booster(model_file=str(modelpath))

    # Predict
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()

    ########################################################
    ########################################################
    ########################################################

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    utils.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = utils.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores

# [Req]


def main(args):
    # [Req]
    cfg = InferConfig()
    model_name = 'Model Name'
    model_params_filepath = 'default_model_params.cfg'
    # Required parameters
    required_parameters = None
    additional_definitions = model_params

    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config=model_params_filepath,
        default_model=None,
        additional_section=model_name,
        additional_definitions=additional_definitions,
        required=required_parameters
    )

    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

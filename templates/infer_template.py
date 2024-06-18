import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# Model-specifc imports
from model_utils.utils import extract_subset_fea

# [Req] Imports from preprocess and train scripts
from lgbm_preprocess_improve import preprocess_params
from lgbm_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
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
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    te_data = pd.read_parquet(Path(params["test_ml_data_dir"])/test_data_fname)

    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # Test data
    xte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep)
    yte = te_data[[params["y_col_name"]]]

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]

    # Load LightGBM
    model = lgb.Booster(model_file=str(modelpath))

    # Predict
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores

# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

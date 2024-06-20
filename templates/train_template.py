import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# Model-specifc imports
from model_utils.utils import extract_subset_fea

# [Req] Imports from preprocess script
from lgbm_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {"name": "learning_rate",
    "type": float,
    "default": 0.1,
    "help": "Learning rate for the optimizer."
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + model_train_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    tr_data = pd.read_parquet(Path(params["train_ml_data_dir"])/train_data_fname)
    vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)

    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # Train data
    xtr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep)
    ytr = tr_data[[params["y_col_name"]]]
    print("xtr:", xtr.shape)
    print("ytr:", ytr.shape)

    # Val data
    xvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep)
    yvl = vl_data[[params["y_col_name"]]]
    print("xvl:", xvl.shape)
    print("yvl:", yvl.shape)

    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------
    # Prepare model and train settings
    ml_init_args = {'n_estimators': 1000, 'max_depth': -1,
                    'learning_rate': params["learning_rate"],
                    'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
    model = lgb.LGBMRegressor(objective='regression', **ml_init_args)

    # Train model
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    ml_fit_args['eval_set'] = (xvl, yvl)
    model.fit(xtr, ytr, **ml_fit_args)

    # Save model
    model.booster_.save_model(str(modelpath))
    del model

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    model = lgb.Booster(model_file=str(modelpath))

    # Compute predictions
    val_pred = model.predict(xvl)
    val_true = yvl.values.squeeze()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores

# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="lgbm_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)
    print("\nFinished model training.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
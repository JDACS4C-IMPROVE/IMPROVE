import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import lightgbm as lgb

# [Req] IMPROVE/CANDLE imports
from improvelib import utils

# Model-specifc imports
from LGBM.model_utils.utils import extract_subset_fea
from lgbm_model_parameters import model_params

from improvelib.applications.drug_response_prediction.config import DRPTrainConfig


filepath = Path(__file__).resolve().parent  # [Req]

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
    # import pdb; pdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    utils.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = utils.build_model_path(
        params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = utils.build_ml_data_name(params, stage="train")
    val_data_fname = utils.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    tr_data = pd.read_parquet(
        Path(params["train_ml_data_dir"])/train_data_fname)
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
    ml_init_args = {"n_estimators": int(params["n_estimators"]),
                    "max_depth": int(params["max_depth"]),
                    "learning_rate": float(params["learning_rate"]),
                    "num_leaves": int(params["num_leaves"]),
                    "n_jobs": 8,
                    "random_state": None}
    model = lgb.LGBMRegressor(objective='regression', **ml_init_args)

    # Train model
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
    ml_fit_args['eval_set'] = (xvl, yvl)
    model.fit(xtr.astype(float), ytr.astype(float), **ml_fit_args)

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
    utils.store_predictions_df(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = utils.compute_performance_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores

# [Req]


def main(args):
    # [Req]
    cfg = DRPTrainConfig()
    model_name = 'LGBM'
    model_params_filepath = 'lgbm_params.txt'
    # Required parameters
    required_parameters = None
    additional_definitions = model_params

    params = cfg.initialize_parameters(
        pathToModelDir=os.path.join(filepath, 'LGBM'),
        default_config=model_params_filepath,
        default_model=None,
        additional_cli_section=model_name,
        additional_definitions=additional_definitions,
        required=required_parameters
    )

    val_scores = run(params)
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

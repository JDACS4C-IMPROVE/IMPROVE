from model_parameters import model_params
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from improvelib.initializer.stage_config import TrainConfig

# [Req] IMPROVE imports
from improvelib.tools import utils

filepath = Path(__file__).resolve().parent  # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Parameter list is required:
# - model_train_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# initialize_parameters().

# 2. Model-specific params (Model: LightGBM)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.


# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
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
    # The files should be an output of your preprocess script
    # ------------------------------------------------------
    tr_data = pd.read_parquet(
        Path(params["train_ml_data_dir"])/train_data_fname)
    vl_data = pd.read_parquet(Path(params["val_ml_data_dir"])/val_data_fname)

    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------

    ########################################################
    ############# Your model's code is here ################
    ########################################################

    # If necessary, extract features and target values in separate variables
    # Train data
    xtr = None  # Features
    ytr = None  # Target values

    # Val data
    xvl = None  # Features
    yvl = None  # Target values

    # Prepare model and train settings
    # Use params dictionary to get training parameters for your model
    # Can be replaced/modified in any way to fit your models
    ml_init_args = None

    stub = {'n_estimators': params['n_estimators'],
            'learning_rate': params["learning_rate"],
            'num_leaves': params['num_leaves'],
            'n_jobs': params['n_jobs'],
            'random_state': None}

    # Initialize model here
    model = None  # lgb.LGBMRegressor(objective='regression', **ml_init_args)

    # Train model
    # Replace the line with your model's training function
    model.fit(xtr, ytr, **ml_init_args)

    # Save model
    # Replace the line with the appropriate code for your model
    model.booster_.save_model(str(modelpath))
    del model

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)

    # Replace the line with the appropriate code for your model
    model = load_model(model_file=str(modelpath))

    # Compute predictions
    # Replace the line with the appropriate code for your model
    val_pred = model.predict(xvl)

    # Shape output of your model to one-dimentional numpy script
    val_true = yvl.values.squeeze()
    ########################################################
    ########################################################
    ########################################################

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
    val_scores = utils.compute_performace_scores(
        params,
        y_true=val_true, y_pred=val_pred, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores

# [Req]


def main(args):
    # [Req]
    cfg = TrainConfig()
    model_name = 'Model Name'
    model_params_filepath = 'default_model_params.cfg'
    # Required parameters
    required_parameters = None
    additional_definitions = model_params

    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
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

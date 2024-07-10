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

# [Req] Imports from preprocess and train scripts
from train_examples import metrics_list
from improvelib.applications.drug_response_prediction.config import DRPInferConfig

filepath = Path(__file__).resolve().parent  # [Req]

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
    # import ipdb; ipdb.set_trace()

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

    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # Test data
    xte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep)
    yte = te_data[[params["y_col_name"]]]
    print("xte:", xte.shape)
    print("yte:", yte.shape)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    params['model_file_format'] = 'pt'
    params['model_file_name'] = 'model'
    params['loss'] = 'r2'
    modelpath = utils.build_model_path(
        params, model_dir=params["model_dir"])  # [Req]

    # Load LightGBM
    model = lgb.Booster(model_file=str(modelpath))

    # Predict
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()

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
    test_scores = utils.compute_performance_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


def main(args):
    # [Req]
    cfg = DRPInferConfig()
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
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

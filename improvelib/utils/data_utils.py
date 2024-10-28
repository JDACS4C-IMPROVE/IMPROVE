"""
This module contains functions for data manipulation and evaluation, including saving dataframes and storing predictions.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def save_stage_ydf(ydf: pd.DataFrame, stage: str, output_dir: str) -> None:
    """Save a subset of y data samples (rows of the input dataframe) to a CSV file.
    The 'stage' refers to one of the three ML stages: 'train', 'val', or 'test'.

    Args:
        ydf (pd.DataFrame): DataFrame containing y data samples.
        stage (str): The stage of the data (e.g., 'train', 'val', 'test').
        output_dir (str): Directory to save the CSV file.
    """
    ydf_fname = f"{stage}_y_data.csv"
    ydf_fpath = Path(output_dir) / ydf_fname
    ydf.to_csv(ydf_fpath, index=False)
    return None


def store_predictions_df(y_pred: np.array,
                         y_col_name: str,
                         stage: str,
                         output_dir: Union[Path, str],
                         input_dir: Optional[str] = None,
                         y_true: Optional[np.array] = None,
                         round_decimals: int = 4) -> None:
    """Save predictions along with the original data in a DataFrame.

    This allows to trace original data evaluated (e.g. drug and cell paris) if
    corresponding dataframe is available, in which case the whole structure as
    well as model predictions are stored. If the dataframe is not available,
    only ground truth and model predictions are stored.

    Args:
        y_pred (np.array): Array of model predictions.
        y_col_name (str): Name of the column in the y data predicted on.
        stage (str): Specify if evaluation is with respect to val or test set.
        output_dir (nion[Path, str]): Directory to write results.
        y_true (Optional[np.array]): Ground truth values.
        input_dir (Optional[str]): Directory where the DataFrame with ground truth is stored.
        round_decimals (int): Number of decimals in output.
    """
    cast_ydata = np.float32

    # Put predictions in a df
    pred_col_name = y_col_name + "_pred" # define colname for predicted values
    pred_df = pd.DataFrame({pred_col_name: y_pred}) # create df
    pred_df = pred_df.astype({pred_col_name: cast_ydata}) # cast
    pred_df = pred_df.round({pred_col_name: round_decimals}) # round decimal

    # Add ground truth values if available to the pred_df
    if y_true is not None:
        # Check that y_true and y_pred dims match
        assert len(y_true) == len(y_pred), f"length mismatch of y_true \
            ({len(y_true)}) and y_pred ({len(y_pred)})"

        true_col_name = y_col_name + "_true"
        pred_df.insert(0, true_col_name, y_true, allow_duplicates=True) # add col to df
        pred_df = pred_df.astype({true_col_name: cast_ydata}) # cast
        pred_df = pred_df.round({true_col_name: round_decimals}) # round decimal

    # ydf refers to a file that can contain metadata of ydata and possibly the
    # ground truth values (e.g., metadata df that contains cancer ids, drug
    # ids, and the true response values)
    ydf_fname = f"{stage}_y_data.csv" # name of ydf if it exists
    ydf_out_fname = ydf_fname.split(".")[0] + "_predicted.csv" # fname for output ydf
    ydf_out_fpath = Path(output_dir) / ydf_out_fname # path for output ydf

    # Attempt to concatenate raw predictions with y dataframe (e.g., metadata
    # df that contains cancer ids, drug ids, and the true response values)
    # Check if ydf exists
    if (input_dir is not None) and (Path(input_dir) / ydf_fname).exists():
        ydf_fpath = Path(input_dir) / ydf_fname
        rsp_df = pd.read_csv(ydf_fpath)
        rsp_df = rsp_df.astype({y_col_name: cast_ydata}) # cast
        rsp_df = rsp_df.round({y_col_name: round_decimals}) # round decimal

        # Check if ground truth is available ydf
        if y_true is not None:
            # Check that ydf and ground truth dims match
            assert len(y_true) == rsp_df.shape[0], f"length mismatch of y_true \
                ({len(y_true)}) and loaded ydf ({ydf_fpath} ==> {rsp_df.shape[0]})"

            if y_col_name in rsp_df.columns:
                v1 = rsp_df[y_col_name].values
                v2 = pred_df[true_col_name].values
                # Check that values of ground truth in ydf and y_true actually match
                assert np.array_equal(v1, v2), "Loaded y data array is not \
                    equal to the true array"

        df = pd.concat([rsp_df, pred_df], axis=1)

    else:
        df = pred_df.copy()

    df.to_csv(ydf_out_fpath, index=False)  # Save predictions df

    return None
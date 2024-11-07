"""Cross-study analysis (CSA) post-processing utilities.

This module provides utilities for post-processing results from cross-study analysis runs,
including runtime analysis and performance metrics calculation.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, sem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from improvelib.metrics import compute_metrics


def apply_decimal_to_dataframe(df: pd.DataFrame, decimal_places: int=4) -> pd.DataFrame:
    """Apply specified decimal places to numeric DataFrame columns.
    
    Args:
        df: DataFrame to modify.
        decimal_places: Desired number of decimal places. Defaults to 4.
        
    Returns:
        pd.DataFrame: Modified DataFrame with specified decimal format applied to numeric columns.
        
    Raises:
        Exception: If error occurs while formatting specific columns.
    """
    for col in df.select_dtypes(include='number').columns:
        try:
            # Round each numeric column to the specified number of decimal places
            df[col] = df[col].round(decimal_places)
        except Exception as e:
            # Log an error message if rounding fails for a column
            print(f"Error formatting column '{col}': {e}")

    return df


def csa_postprocess(res_dir_path: Union[str, Path],
                    model_name: str,
                    y_col_name: str,
                    metric_type: str="regression",
                    decimal_places: int=4,
                    outdir: str="./",
                    verbose: bool=False) -> pd.DataFrame:
    """Generate cross-study analysis tables and figures.
    
    Args:
        res_dir_path: Path to cross-study results directory.
        model_name: Name of the model (e.g., GraphDRP, IGTD).
        y_col_name: Name of prediction variable.
        metric_type: Type of metrics to compute. Defaults to "regression".
        decimal_places: Number of decimal places for results. Defaults to 4.
        outdir: Path to save results. Defaults to "./".
        verbose: Whether to print detailed output. Defaults to False.
        
    Returns:
        pd.DataFrame: Performance scores DataFrame containing:
            - met: Metric name
            - split: Split number
            - value: Score value
            - src: Source dataset name
            - trg: Target dataset name
            - model: Model name
        
    Raises:
        FileNotFoundError: If prediction files are not found.
        Exception: If unexpected error occurs during processing.
    """
    infer_dir_name = "infer"
    infer_dir_path = res_dir_path / infer_dir_name
    # List all directories matching the pattern "*-*" in the infer directory
    dirs = sorted(list(infer_dir_path.glob("*-*")))

    os.makedirs(outdir, exist_ok=True)

    def calc_mae(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """Calculates the Mean Absolute Error (MAE) between true and predicted values.
        
        Args:
            y_true: Array of true values.
            y_pred: Array of predicted values.

        Returns:
            float: The mean absolute error between `y_true` and `y_pred`.
        
        Raises:
            ValueError: If `y_true` and `y_pred` have different lengths.
        """
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)

    def calc_r2(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """Calculates the R-squared (R²) score between true and predicted values.
        
        Args:
            y_true: Array of true values.
            y_pred: Array of predicted values.

        Returns:
            float: The R-squared score between `y_true` and `y_pred`.
        
        Raises:
            ValueError: If `y_true` and `y_pred` have different lengths.
        """
        return r2_score(y_true=y_true, y_pred=y_pred)

    def calc_pcc(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """Calculates the Pearson Correlation Coefficient between true and predicted values.
        
        Args:
            y_true: Array of true values.
            y_pred: Array of predicted values.

        Returns:
            float: The Pearson correlation coefficient between `y_true` and `y_pred`.
        
        Raises:
            ValueError: If `y_true` and `y_pred` have different lengths.
        """
        return pearsonr(y_true, y_pred)[0]

    def calc_scc(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """Calculates the Spearman Correlation Coefficient between true and predicted values.
        
        Args:
            y_true: Array of true values.
            y_pred: Array of predicted values.

        Returns:
            float: The Spearman correlation coefficient between `y_true` and `y_pred`.
        
        Raises:
            ValueError: If `y_true` and `y_pred` have different lengths.
        """
        return spearmanr(y_true, y_pred)[0]

    scores_names = {"mae": calc_mae,
                    "r2": calc_r2,
                    "pcc": calc_pcc,
                    "scc": calc_scc}

    preds_file_name = "test_y_data_predicted.csv"
    sep = ','
    scores_fpath = outdir / "all_scores.csv"
    missing_pred_files = []

    if scores_fpath.exists(): 
        print("Load scores")
        scores = pd.read_csv(scores_fpath, sep=sep)
    else:
        print("Calc scores")
        dfs = []
        for dir_path in dirs:
            print("Experiment:", dir_path)
            src = str(dir_path.name).split("-")[0]
            trg = str(dir_path.name).split("-")[1]
            split_dirs = sorted(list((dir_path).glob(f"split_*")))

            jj = {}

            for split_dir in split_dirs:
                preds_file_path = split_dir / preds_file_name
                try:
                    # Read predicted and true values from CSV
                    preds = pd.read_csv(preds_file_path, sep=sep)
                    y_true = preds[f"{y_col_name}_true"].values
                    y_pred = preds[f"{y_col_name}_pred"].values
                    # Compute metrics for the current split
                    sc = compute_metrics(y_true, y_pred, metric_type=metric_type)
                    split = int(split_dir.name.split("split_")[1])
                    jj[split] = sc
                    del preds, y_true, y_pred, sc, split

                except FileNotFoundError:
                    print(f"Error: File not found! {preds_file_path}")
                    missing_pred_files.append(preds_file_path)

                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

            df = pd.DataFrame(jj)
            df = df.stack().reset_index()
            df.columns = ['met', 'split', 'value']
            df['src'] = src
            df['trg'] = trg
            if df.empty is False:
                dfs.append(df)

        scores = pd.concat(dfs, axis=0)
        scores['model'] = model_name
        scores.to_csv(outdir / "all_scores.csv", index=False)
        del dfs

        if len(missing_pred_files) > 0:
            with open(f"{outdir}/missing_pred_files.txt", "w") as f:
                for line in missing_pred_files:
                    line = 'infer' + str(line).split('infer')[1]
                    f.write(line + "\n")

    # Calculate mean and standard deviation for each metric
    sc_mean = scores.groupby(["met", "src", "trg"])["value"].mean().reset_index()
    sc_std = scores.groupby(["met", "src", "trg"])["value"].std().reset_index()

    mean_tb = {}
    std_tb = {}
    for met in scores.met.unique():
        df = scores[scores.met == met]
        df['model'] = model_name
        df.to_csv(outdir / f"{met}_scores.csv", index=True)
        mean = df.groupby(["src", "trg"])["value"].mean()
        mean = mean.unstack()
        mean = apply_decimal_to_dataframe(mean, decimal_places)
        mean.to_csv(outdir / f"{met}_mean_csa_table.csv", index=True)
        print(f"{met} mean:\n{mean}")
        mean_tb[met] = mean
        std = df.groupby(["src", "trg"])["value"].std()
        std = std.unstack()
        std.to_csv(outdir / f"{met}_std_csa_table.csv", index=True)
        print(f"{met} std:\n{std}")
        std_tb[met] = std

    # Separate within-dataset and cross-dataset results
    df_on = scores[scores.src == scores.trg].reset_index()
    on_mean = df_on.groupby(["met"])["value"].mean().reset_index().rename(columns={"value": "mean"})
    on_std = df_on.groupby(["met"])["value"].std().reset_index().rename(columns={"value": "std"})
    on = on_mean.merge(on_std, on="met", how="inner")
    on["summary"] = "within"

    df_off = scores[scores.src != scores.trg]
    off_mean = df_off.groupby(["met"])["value"].mean().reset_index().rename(columns={"value": "mean"})
    off_std = df_off.groupby(["met"])["value"].std().reset_index().rename(columns={"value": "std"})
    off = off_mean.merge(off_std, on="met", how="inner")
    off["summary"] = "cross"

    if verbose:
        print(f"On-diag mean:\n{on_mean}")
        print(f"On-diag std: \n{on_std}")
        print(f"Off-diag mean:\n{off_mean}")
        print(f"Off-diag std: \n{off_std}")

    df = pd.concat([on, off], axis=0).sort_values("met")
    df['model'] = model_name
    df.to_csv(outdir / "densed_csa_table.csv", index=False)
    print(f"Densed CSA table:\n{df}")
    return scores


def runtime_analysis(res_dir_path: Union[str, Path],
                     stage_dir_name: str,
                     model_name: str,
                     res_fname: str='runtime.json',
                     decimal_places: int=4,
                     verbose: bool=False) -> Union[pd.DataFrame, None]:
    """Analyze runtime performance for different stages.
    
    Args:
        res_dir_path: Output directory containing all CSA results.
        stage_dir_name: Directory containing specific stage results (e.g., ml_data, models, infer).
        model_name: Name of the model being analyzed.
        res_fname: File name containing raw runtime results. Defaults to 'runtime.json'.
        decimal_places: Number of decimal places for results. Defaults to 4.
        verbose: Whether to print detailed output. Defaults to False.
        
    Returns:
        pd.DataFrame or None: Aggregated runtime results containing:
            - src: Source dataset name
            - trg: Target dataset name
            - split: Split number
            - hours: Runtime hours
            - minutes: Runtime minutes
            - tot_mins: Total runtime in minutes
            - model: Model name
            Returns None if no results are available.
            
    Warnings:
        UserWarning: If runtime files are not found.
    """
    stage_dir_path = Path(res_dir_path) / stage_dir_name
    stage_dirs = sorted(list(stage_dir_path.glob("*")))

    missing_files = []
    jj = []

    for dir_path in stage_dirs:
        dir_name = str(dir_path.name).split("-")
        src = dir_name[0]
        trg = dir_name[1] if len(dir_name) > 1 else 'NA'

        split_dirs = sorted(list((dir_path).glob(f"split_*")))

        for split_dir in split_dirs:
            runtime_file_path = split_dir / res_fname
            try:
                # Load runtime data from JSON file
                with open(runtime_file_path, 'r') as file:
                    rr = json.load(file)
                rr['src'] = src
                rr['trg'] = trg
                split = int(split_dir.name.split("split_")[1])
                rr['split'] = split
                jj.append(rr)
            except FileNotFoundError:
                warnings.warn(f"File not found! {runtime_file_path}", UserWarning)
                missing_files.append(runtime_file_path)

    df = None
    if len(jj) > 0:
        df = pd.DataFrame(jj)
        df = df.replace(to_replace='NA', value=None)
        # Calculate total runtime in minutes
        df['tot_mins'] = df['hours'] * 60 + df['minutes']
        df['model'] = model_name
    return df


def plot_color_coded_csa_table(df: pd.DataFrame,
                               filepath: str="./",
                               title: str=None):
    """Creates and saves a color-coded table as a heatmap figure.
    
    Args:
        df: DataFrame containing the data to visualize.
        filepath: Path where the figure will be saved. Defaults to "./".
        title: Title for the plot. Defaults to None.
        
    Returns:
        None: Saves the plot as an image file to the specified filepath.
    """
    df = pd.DataFrame(df)
    df.set_index('src', inplace=True)

    # Create a diverging color palette from red to green
    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True).reversed()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, annot=True, cmap=cmap, center=0, cbar=False,
                     linewidths=0.5, linecolor='gray', fmt=".2f")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title(title)

    # Save the heatmap as an image file
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

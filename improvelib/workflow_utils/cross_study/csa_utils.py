""" Postprocessing results from Cross-Study Analysis (CSA) runs. """

import os
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, sem
from typing import Optional

# from improve.metrics import compute_metrics
from improvelib.metrics import compute_metrics


def splits_generator():
    """ Generates data splits for cross-study analysis. """
    return None


def csa_postprocess(res_dir_path,
                    model_name: str,
                    y_col_name: str,
                    round_digits: int=3,
                    outdir: str="./",
                    verbose: bool=False):
    """ Generates cross-study analysis tables and a figure.

    Args:
        res_dir_path: full path to the cross-study results dir
        model_name: name of the model (e.g., GraphDRP, IGTD)
        y_col_name: prediction variable
        outdir: full path to save the csa post-processing results

    Return:
        performance scores for all source-target pairs and splits
    """
    infer_dir_name = "infer"
    infer_dir_path = res_dir_path/infer_dir_name
    dirs = list(infer_dir_path.glob("*-*")); print(dirs)
    # print(split_files)

    os.makedirs(outdir, exist_ok=True)

    data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"] # TODO extract this from res_dir

    def calc_mae(y_true, y_pred):
        return sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

    def calc_r2(y_true, y_pred):
        return sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)

    def calc_pcc(y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]

    def calc_scc(y_true, y_pred):
        return spearmanr(y_true, y_pred)[0]

    scores_names = {"mae": calc_mae,
                    "r2": calc_r2,
                    "pcc": calc_pcc,
                    "scc": calc_scc}

    # ====================
    # Aggregate raw scores
    # ====================

    preds_file_name = "test_y_data_predicted.csv"
    # scores_file_name = "test_scores.json"
    metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

    sep = ','
    scores_fpath = outdir / "all_scores.csv"
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
            split_dirs = list((dir_path).glob(f"split_*"))

            jj = {}  # dict (key: split id, value: dict of scores)

            for split_dir in split_dirs:
                # Load preds
                preds_file_path = split_dir / preds_file_name
                preds = pd.read_csv(preds_file_path, sep=sep)

                # Compute scores
                y_true = preds[f"{y_col_name}_true"].values
                y_pred = preds[f"{y_col_name}_pred"].values
                sc = compute_metrics(y_true, y_pred, metrics_list)

                split = int(split_dir.name.split("split_")[1])
                jj[split] = sc
                # df = pd.DataFrame(jj)
                # df = df.T.reset_index().rename(columns={"index": "split"})

            # Convert dict to df, and aggregate dfs
            df = pd.DataFrame(jj)
            df = df.stack().reset_index()
            df.columns = ['met', 'split', 'value']
            df['src'] = src
            df['trg'] = trg
            dfs.append(df)

        # Concat dfs and save
        scores = pd.concat(dfs, axis=0)
        scores.to_csv(outdir / "all_scores.csv", index=False)
        del dfs

    # Average across splits
    sc_mean = scores.groupby(["met", "src", "trg"])["value"].mean().reset_index()
    sc_std = scores.groupby(["met", "src", "trg"])["value"].std().reset_index()

    # Generate csa table
    mean_tb = {}
    std_tb = {}
    for met in scores.met.unique():
        df = scores[scores.met == met]
        # df = df.sort_values(["src", "trg", "met", "split"])
        df.to_csv(outdir / f"{met}_scores.csv", index=True)
        # Mean
        mean = df.groupby(["src", "trg"])["value"].mean()
        mean = mean.unstack()
        mean.to_csv(outdir / f"{met}_mean_csa_table.csv", index=True)
        print(f"{met} mean:\n{mean}")
        mean_tb[met] = mean
        # Std
        std = df.groupby(["src", "trg"])["value"].std()
        std = std.unstack()
        std.to_csv(outdir / f"{met}_std_csa_table.csv", index=True)
        print(f"{met} std:\n{std}")
        std_tb[met] = std

    # Quick test
    # met="mse"; src="CCLE"; trg="GDSCv1" 
    # print(f"src: {src}; trg: {trg}; met: {met}; mean: {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.mean()}")
    # print(f"src: {src}; trg: {trg}; met: {met}; std:  {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.std()}")
    # met="mse"; src="CCLE"; trg="GDSCv2" 
    # print(f"src: {src}; trg: {trg}; met: {met}; mean: {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.mean()}")
    # print(f"src: {src}; trg: {trg}; met: {met}; std:  {scores[(scores.met==met) & (scores.src==src) & (scores.trg==trg)].value.std()}")

    # Generate densed csa table
    # breakpoint()
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

    # Combine dfs
    # breakpoint()
    df = pd.concat([on, off], axis=0).sort_values("met")
    df.to_csv(outdir / "densed_csa_table.csv", index=False)
    print(f"Densed CSA table:\n{df}")
    return scores


def plot_color_coded_csa_table(df: pd.DataFrame, filepath: str="./", title: str=None):
    """
    Creates a color-coded table with shades of red and green based on the values and saves it as a figure.

    Args:
        data (dict): Dictionary containing the data for the table.
        filename (str): The filename for the saved figure.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a DataFrame
    df = pd.DataFrame(df)
    df.set_index('src', inplace=True)

    # Create a color map from red to green
    # https://seaborn.pydata.org/tutorial/color_palettes.html
    # cmap = sns.diverging_palette(220, 20, as_cmap=True).reversed()
    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True).reversed()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, annot=True, cmap=cmap, center=0, cbar=False,
                     linewidths=0.5, linecolor='gray', fmt=".2f")

    # Set the labels and title
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Set the title
    plt.title(title)

    # Save the plot
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

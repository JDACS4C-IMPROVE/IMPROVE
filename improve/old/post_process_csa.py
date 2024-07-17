import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, sem

from model_utils.utils import Timer
from improve.metrics import compute_metrics

fdir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', required=False, default='auc', type=str,
                    help='Dir containing the results.')
parser.add_argument('--model_name', required=False, default='LGBM', type=str,
                    help='Name of the model.')
parser.add_argument('--y_col_name', required=False, default='auc', type=str,
                    help='Y col name.')
args = parser.parse_args()

res_dir = args.res_dir
model_name = args.model_name
y_col_name = args.y_col_name

main_dir_path = Path(f"/lambda_stor/data/apartin/projects/IMPROVE/pan-models/{model_name}")
infer_dir_name = "infer"
# infer_dir_path = main_dir_path/y_col_name/infer_dir_name
infer_dir_path = main_dir_path/res_dir/infer_dir_name
dirs = list(infer_dir_path.glob("*-*")); print(dirs)
# print(split_files)

# model_name = args.model_name
# model_name = "GraphDRP_01"
# model_name = "GraphDRP_02"
# model_name = "GraphDRP_03"

# datadir = fdir/f"results.{model_name}"
# outdir = fdir/f"scores.{model_name}"
# os.makedirs(outdir, exist_ok=True)

# outdir = fdir/f"../res.csa.{model_name}"
outdir = fdir/f"../res.csa.{model_name}.{res_dir}"
os.makedirs(outdir, exist_ok=True)

data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]
trg_name = "AUC"
round_digits = 3

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
scores_fpath = outdir/"all_scores.csv"
timer = Timer()
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

        # import ipdb; ipdb. set_trace()
        for split_dir in split_dirs:
            # Load preds
            preds_file_path = split_dir/preds_file_name
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
    # scores = scores.sort_values(["src", "trg", "met", "split"])
    scores.to_csv(outdir/"all_scores.csv", index=False)

timer.display_timer()

# Average across splits
sc_mean = scores.groupby(["met", "src", "trg"])["value"].mean().reset_index()
sc_std = scores.groupby(["met", "src", "trg"])["value"].std().reset_index()

# Generate csa table
# TODO
breakpoint()
mean_tb = {}
std_tb = {}
for met in scores.met.unique():
    df = scores[scores.met == met]
    # df = df.sort_values(["src", "trg", "met", "split"])
    df.to_csv(outdir/f"{met}_scores.csv", index=True)
    # Mean
    mean = df.groupby(["src", "trg"])["value"].mean()
    mean = mean.unstack()
    mean.to_csv(outdir/f"{met}_mean_csa_table.csv", index=True)
    print(f"{met} mean:\n{mean}")
    mean_tb[met] = mean
    # Std
    std = df.groupby(["src", "trg"])["value"].std()
    std = std.unstack()
    std.to_csv(outdir/f"{met}_std_csa_table.csv", index=True)
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
breakpoint()
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

print(f"On-diag mean:\n{on_mean}")
print(f"On-diag std: \n{on_std}")

print(f"Off-diag mean:\n{off_mean}")
print(f"Off-diag std: \n{off_std}")

# Combine dfs
breakpoint()
df = pd.concat([on, off], axis=0).sort_values("met")
df.to_csv(outdir/"densed_csa_table.csv", index=False)
print(f"Densed CSA table:\n{df}")


# # ---------------------
# # Data source study
# for sc_name, sc_func in scores_names.items():
#     print("\nMetric:", sc_name)
#     for src in data_sources:
#         print("\n\tSource study:", src)
#         # resdir = fdir/f"results.csa.{src}"
#         # resdir = datadir/f"results.csa.{src}"
#         resdir = datadir
#         scores = {}

#         # Data test study
#         for trg in data_sources:
#             print("\tTraget study:", trg)
#             if trg not in scores:
#                 # scores[trg] = {sc: [] for sc in scores_names}
#                 scores[trg] = []

#             # Data split
#             for split in range(10):
#                 fname = f"{src}_{trg}_split_{split}.csv"
#                 df = pd.read_csv(resdir/fname, sep=",")
#                 # df = pd.read_csv(resdir/fname, sep="\t")

#                 # for sc_name, sc_func in scores_names.items():
#                 y_true = df["True"].values
#                 y_pred = df["Pred"].values
#                 sc_value = sc_func(y_true=y_true, y_pred=y_pred)
#                 # scores[trg][sc_name].append(sc_value)
#                 scores[trg].append(sc_value)

#         with open(outdir/f"{sc_name}_{src}_scores_raw.json", "w") as json_file:
#             json.dump(scores, json_file)
# del scores
# # ---------------------


# # ====================
# # Generate tables
# # ====================
# # Data source study
# for sc_name in scores_names.keys():
#     print("\nMetric:", sc_name)

#     mean_df = {}
#     err_df = {}
#     for src in data_sources:
#         print("\tSource study:", src)

#         with open(outdir/f"{sc_name}_{src}_scores_raw.json") as json_file:
#             mean_scores = json.load(json_file)
#         err_scores = mean_scores.copy()

#         for trg in data_sources:
#             mean_scores[trg] = round(np.mean(mean_scores[trg]), round_digits)
#             err_scores[trg] = round(sem(err_scores[trg]), round_digits)

#         mean_df[src] = mean_scores
#         err_df[src] = err_scores

#     mean_df = pd.DataFrame(mean_df)
#     err_df = pd.DataFrame(err_df)
#     mean_df.to_csv(outdir/f"{sc_name}_mean_table.csv", index=True)
#     err_df.to_csv(outdir/f"{sc_name}_err_table.csv", index=True)


# # ====================
# # Summary table
# # ====================
# # Data source study
# sc_df = []
# cols = ["Metric", "Diag", "Off-diag", "Diag_std", "Off-diag_std"]
# for i, sc_name in enumerate(scores_names):
#     # print("\nMetric:", sc_name)
#     sc_item = {}
#     sc_item["Metric"] = sc_name.upper()

#     mean_df = pd.read_csv(outdir/f"{sc_name}_mean_table.csv")
#     n = mean_df.shape[0]

#     # Diag
#     vv = np.diag(mean_df.iloc[:, 1:].values)
#     sc_item["Diag"] = np.round(sum(vv)/n, 3)
#     sc_item["Diag_std"] = np.round(np.std(vv), 3)

#     # Off-diag
#     vv = mean_df.iloc[:, 1:].values
#     np.fill_diagonal(vv, 0)
#     sc_item["Off-diag"] = np.round(sum(np.ravel(vv)) / (n*n - n), 3)
#     sc_item["Off-diag_std"] = np.round(np.std(np.ravel(vv)), 3)

#     for ii, dname in enumerate(mean_df.iloc[:, 0].values):
#         dname = dname.upper()
#         sc_item[dname] = np.round(sum(vv[ii, :] / (n - 1)), 3)
#         if i == 0:
#             cols.append(dname)

#     sc_df.append(sc_item)

# sc_df = pd.DataFrame(sc_df, columns=cols)
# sc_df.to_csv(outdir/f"summary_table.csv", index=False)
# print(sc_df)

print("Finished all")

"""
python csa_postproc.py --res_dir res.csa --model_name GraphDRP --y_col_name auc
"""
import argparse
import os
from pathlib import Path

import pandas as pd

def csa_get_predictions(res_dir_path, model_name, y_col_name, outdir="./"):

    infer_dir_name = "infer"
    infer_dir_path = res_dir_path/infer_dir_name
    dirs = sorted(list(infer_dir_path.glob("*-*")))

    os.makedirs(outdir, exist_ok=True)

    # ====================
    # Aggregate raw scores
    # ====================

    preds_file_name = "test_y_data_predicted.csv"
    sep = ','

    missing_pred_files = []

    dfs = []
    for dir_path in dirs:
        print("Experiment:", dir_path)
        src = str(dir_path.name).split("-")[0]
        trg = str(dir_path.name).split("-")[1]
        split_dirs = sorted(list((dir_path).glob(f"split_*")))

        for split_dir in split_dirs:
            preds_file_path = split_dir / preds_file_name
            try:
                columns_to_load = ["improve_sample_id", "improve_chem_id", f"{y_col_name}_true", f"{y_col_name}_pred"]
                preds = pd.read_csv(preds_file_path, sep=sep,
                                    usecols=columns_to_load)
                split = int(split_dir.name.split("split_")[1])
                preds['source'] = src
                preds['target'] = trg
                preds['split'] = split

                dfs = dfs + [preds]
                # Clean
                del preds, split

            except FileNotFoundError:
                print(f"Error: File not found! {preds_file_path}")
                missing_pred_files.append(preds_file_path)

            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    # Concat dfs and save
    scores = pd.concat(dfs, axis=0)
    scores['model'] = model_name
    scores.to_parquet(outdir + "/" + model_name + "_all_predictions.parquet", index=False)
    del dfs

    if len(missing_pred_files) > 0:
        with open(f"{outdir}/missing_pred_files_predictions.txt", "w") as f:
            for line in missing_pred_files:
                line = 'infer' + str(line).split('infer')[1]
                f.write(line + "\n")

    return scores




filepath = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir',
                    type=str,
                    required=True,
                    help='Dir containing the results.')
parser.add_argument('--model_name',
                    type=str,
                    required=True,
                    help='Name of the model (e.g., GraphDRP, DeepCDR).')
parser.add_argument('--y_col_name',
                    type=str,
                    default='auc',
                    required=False,
                    help='Y col name.')
parser.add_argument('--outdir',
                    type=str,
                    default=None,
                    required=False,
                    help='Dir to save post-processing results.')
args = parser.parse_args()

# Args
res_dir = args.res_dir
model_name = args.model_name
y_col_name = args.y_col_name

res_dir_path = Path(res_dir).resolve()  # absolute path to CSA result dir

# Outdir
if args.outdir is None:
    outdir = res_dir_path.parent / f'postproc.csa.{model_name}.{res_dir_path.name}'
else:
    outdir = args.outdir
os.makedirs(outdir, exist_ok=True)


# --------------------------------
# CSA runtime performance analysis
# --------------------------------
# Define a mapping of stage_dir_name to stage
stage_mapping = {
    'ml_data': 'preprocess',
    'models': 'train',
    'infer': 'infer'
}

# -----------------------------------
# CSA prediction performance analysis
# -----------------------------------
scores = csa_get_predictions(res_dir_path,
                         model_name,
                         y_col_name,
                         outdir=outdir)

print('\nFinished cross-study post-processing.')


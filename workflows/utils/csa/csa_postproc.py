"""
python csa_postproc.py --res_dir res.csa --model_name GraphDRP --y_col_name auc
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# IMPROVE imports
# from improvelib.workflow_utils.cross_study.csa_utils import (
# from workflows.utils.csa.csa_utils import (
from csa_utils import (
    runtime_analysis,
    csa_postprocess,
    plot_color_coded_csa_table
)


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

times = []
for stage_dir_name, stage_name in stage_mapping.items():
    df = runtime_analysis(res_dir_path,
                          stage_dir_name,
                          model_name,
                          decimal_places=4)
    if df is not None:
        df['stage'] = stage_name
        times.append(df)

if len(times) > 0:
    times = pd.concat(times, axis=0)
    times.to_csv(outdir / f"runtimes.csv", index=False)


# -----------------------------------
# CSA prediction performance analysis
# -----------------------------------
scores = csa_postprocess(res_dir_path,
                         model_name,
                         y_col_name,
                         decimal_places=4,
                         outdir=outdir)

print('\nFinished cross-study post-processing.')

#!/bin/bash

export IMPROVE_BENCHMARK_DIR="/nfs/lambda_stor_01/data/apartin/projects/IMPROVE/pan-models/IMPROVE/csa_data/raw_data"
python3 preprocessing_examples.py --input_dir $IMPROVE_BENCHMARK_DIR

python3 train_examples.py --data_format='.parquet' --y_col_name='auc' --y_data_suffix='example'

python3 infer_examples.py --data_format='.parquet' --y_col_name='auc' --y_data_suffix='example' \
    --model_file_format='pt' --model_file_name='model' --y_data_preds_suffix='example' --loss='r2' --json_scores_suffix='results'

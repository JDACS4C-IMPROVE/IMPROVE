#!/bin/bash

# Example: bash gen_lc_splits.sh ../../../csa_data/raw_data/y_data/response.tsv ../../../csa_data/raw_data/splits

# LC_SIZES=7
LC_SIZES=10
# LC_SIZES=12

echo "LC sizes: $LC_SIZES"

# dpath=../../../csa_data/raw_data/y_data/response.tsv
# spath=../../../csa_data/raw_data/splits

dpath=$1
spath=$2

python generate_lc_splits.py \
    --data_file_path $dpath \
    --splits_dir $spath \
    --lc_sizes $LC_SIZES \
    --min_size 1024 \
    --lc_step_scale log

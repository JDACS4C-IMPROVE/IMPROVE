#!/bin/bash

# bash subprocess_train.sh ml_data/CCLE-CCLE/split_0 ml_data/CCLE-CCLE/split_0 out_model/CCLE/split_0
# CUDA_VISIBLE_DEVICES=5 bash subprocess_train.sh ml_data/CCLE-CCLE/split_0 ml_data/CCLE-CCLE/split_0 out_model/CCLE/split_0

# Need to comment this when using ' eval "$(conda shell.bash hook)" '
# set -e

# Activate conda env for model using "conda activate myenv"
# https://saturncloud.io/blog/activating-conda-environments-from-scripts-a-guide-for-data-scientists
# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# This doesn't work w/o eval "$(conda shell.bash hook)"
#CONDA_ENV=$PathDSP_env
#echo "Allow conda commands in shell script by running 'conda shell.bash hook'"
#eval "$(conda shell.bash hook)"
#echo "Activated conda commands in shell script"
#conda activate $CONDA_ENV
#source activate $CONDA_ENV
#conda_path=${dirname `which conda`}
#source $conda_path/activate $CONDA_ENV
#source activate $CONDA_ENV
#echo "Activated conda env $CONDA_ENV"

train_ml_data_dir=$1
val_ml_data_dir=$2
model_outdir=$3
learning_rate=$4
batch_size=$5
dropout=$6
CUDA_VISIBLE_DEVICES=$7

echo "train_ml_data_dir:    $train_ml_data_dir"
echo "val_ml_data_dir:      $val_ml_data_dir"
echo "model_outdir:         $model_outdir"
echo "learning_rate:      $learning_rate"
echo "batch_size:      $batch_size"
echo "dropout:      $dropout"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# epochs=10
epochs=50
# epochs=50

# All train outputs are saved in params["model_outdir"]
#CUDA_VISIBLE_DEVICES=6,7 python PathDSP_train_improve.py \
#CUDA_VISIBLE_DEVICES=5
#CUDA_VISIBLE_DEVICES=6,7
# python PathDSP_train_improve.py \
#     --train_ml_data_dir $train_ml_data_dir \
#     --val_ml_data_dir $val_ml_data_dir \
#     --model_outdir $model_outdir \
#     --epochs $epochs

#conda deactivate
#source $conda_path/deactivate
#echo "Deactivated conda env $CONDA_ENV"

echo "train using singularity container"
singularity exec --nv --bind ${IMPROVE_DATA_DIR}:/candle_data_dir $PathDSP_sif train.sh ${CUDA_VISIBLE_DEVICES} /candle_data_dir --train_ml_data_dir /candle_data_dir/$train_ml_data_dir --val_ml_data_dir /candle_data_dir/$val_ml_data_dir/ --model_outdir /candle_data_dir/$model_outdir/ --epochs $epochs --learning_rate $learning_rate --batch_size $batch_size --dropout $dropout


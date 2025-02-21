#!/bin/bash

# bash subprocess_train.sh ml_data/CCLE-CCLE/split_0 ml_data/CCLE-CCLE/split_0 out_model/CCLE/split_0
# CUDA_VISIBLE_DEVICES=5 bash subprocess_train.sh ml_data/CCLE-CCLE/split_0 ml_data/CCLE-CCLE/split_0 out_model/CCLE/split_0

# Need to comment this when using ' eval "$(conda shell.bash hook)" '
# set -e

# Activate conda env for model using "conda activate myenv"
# https://saturncloud.io/blog/activating-conda-environments-from-scripts-a-guide-for-data-scientists
# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# This doesn't work w/o eval "$(conda shell.bash hook)"
CONDA_ENV=$1
echo "Activated conda commands in shell script"
conda_path=$(dirname $(dirname $(which conda)))
source $conda_path/bin/activate $CONDA_ENV
echo "Activated conda env $CONDA_ENV"

# get mandatory arguments
SCRIPT=$2
input_dir=$3
output_dir=$4
CUDA_VISIBLE_DEVICES=$5

command="python $SCRIPT --input_dir $input_dir --output_dir $output_dir "


# append hyperparameter arguments to python call
for i in $(seq 6 $#)
do
    if [ $(($i % 2)) == 0 ]; then
        command="${command} --${!i}"
    else
        command="${command} ${!i}"
    fi
done


echo "command: $command"

# run python script
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $command


source $conda_path/bin/deactivate
echo "Deactivated conda env $CONDA_ENV"

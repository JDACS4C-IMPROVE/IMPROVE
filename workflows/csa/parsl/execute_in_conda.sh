#!/bin/bash

model_env=$1 ; shift
script=$1 ; shift
conda_path=$(dirname $(dirname $(which conda)))
source $conda_path/bin/activate $model_env
CMD="python ${script} $@" #TODO: Run setup_improve.sh for the model here as well.

echo "running command ${CMD}"

$CMD

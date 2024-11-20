#!/bin/bash

model_env=$1 ; shift
script=$1 ; shift
conda_path=$(dirname $(dirname $(which conda)))
source $conda_path/bin/activate $model_env
CMD="python ${script} $@"

echo "running command ${CMD}"

$CMD

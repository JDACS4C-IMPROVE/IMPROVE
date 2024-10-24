#!/bin/bash --login
# Navigate to the IMPROVE repo dir
# Run it like this: source ./setup_improve.sh

# set -e

# Get IMPROVE dir path
improve_lib_path=$PWD
echo "IMPROVE path: $improve_lib_path"

# Checkout branch
improve_branch="develop"
git checkout $improve_branch

# Env var PYTHOPATH
export PYTHONPATH=$PYTHONPATH:$improve_lib_path

echo
echo "PYTHONPATH: $PYTHONPATH"

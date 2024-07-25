#!/bin/bash --login
# Navigate to the dir with IMPROVE repo
# Run it like this: source ./setup_improve.sh

# set -e

# Download data (if needed)
data_dir="csa_data"
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    source download_csa.sh
else
    echo "CSA data folder already exists"
fi

improve_lib_path=$PWD

# Env var PYTHOPATH
export PYTHONPATH=$PYTHONPATH:$improve_lib_path

echo
echo "PYTHONPATH: $PYTHONPATH"

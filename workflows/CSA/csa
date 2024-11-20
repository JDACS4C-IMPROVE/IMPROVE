#!/bin/bash

# Check if at least two positional arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <preprocess|workflow> <arg> ..."
    exit 1
fi

# Check if python or python3 is installed on the system, map to python
if command -v python3 &> /dev/null; then
    echo "Python3 is installed on the system."
fi
if command -v python &> /dev/null; then
    echo "Python is installed on the system."
    echo "Using python for execution : $(command -v python)"
else
    echo "Python is not installed on the system."
    exit 1
fi


# Extract the first positional argument to decide action
ACTION=$1
SECOND_ARG=$2

export IMPROVE_LOG_LEVEL=${IMPROVE_LOG_LEVEL:-INFO}

# Shift arguments so additional arguments are passed on
shift 1

# Handle the "preprocess" argument
if [ "$ACTION" == "preprocess" ]; then
    # Run preprocess.py with any additional arguments
    python "$(dirname "$0")/preprocess.py" "$@"

# Handle the "workflow" argument
elif [ "$ACTION" == "workflow" ]; then
    # Add workflow execution logic here
    echo "Workflow argument received. You can implement workflow logic here."
    python "$(dirname "$0")/train_and_infer.py" "$@"

else
    echo "Invalid action: $ACTION"
    echo "Usage: $0 <preprocess|workflow> <arg>"
    exit 1
fi
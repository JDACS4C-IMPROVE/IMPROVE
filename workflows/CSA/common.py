import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))


prefix = f"START=$(date +%s) ; echo Start:\t$START "
suffix = f"STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1"



# Load conda environment if provided
def get_conda_env(conda_env):
    if conda_env:
        logger.debug(f"Setting conda environment: {conda_env}")
        conda= f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {conda_env} "
    else:
        logger.debug("No conda environment provided")
        conda = "echo no conda env provided"
    return conda

# Make command line call for bash_app
def make_call(cli, conda_env = None , prefix = prefix, suffix = suffix):
    call=[]
    
    if prefix:
        call.append(prefix)
    
    if conda_env:
        call.append(get_conda_env(conda_env))
    
    call.append(" ".join(cli))
    
    if suffix:
        call.append(suffix)

    command = " ;".join(call)
    logger.debug(f"Command line call: {command}")
    return command

# Create path
def make_path(base_dir=None, stage=None, model=None, source_dataset=None, target_dataset=None, split=None, make_dir=True):

    sections = [base_dir, stage, model]

    if not all(sections):
        raise ValueError("Missing required parameters: base_dir, stage, model")
    
    if source_dataset and target_dataset:
        sections.append("-".join([source_dataset, target_dataset]))
    elif source_dataset:
        sections.append(source_dataset)
    elif target_dataset:
        sections.append(target_dataset)
    else:
        raise ValueError("Missing required parameters: source_dataset, target_dataset")
    
    if split:
        sections.append(split)
    

    # Create directory paths
    path = os.path.join( *sections )
    
    # Create the output directory
    if not os.path.exists(path):
        if make_dir:
            logger.debug(f"Creating path: {path}")
            os.makedirs(path, exist_ok=True)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    return path
import sys
import json
import os
import parsl
from parsl import bash_app
import subprocess
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from time import time
from typing import Sequence, Tuple, Union
from pathlib import Path
import logging
import shutil


import csa_params_def as CSA
import improvelib.utils as frm
import improvelib.config.csa as csa

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))


@bash_app
def preprocess(input_file: str, output_file: str, script: str, stdout: str, stderr: str , 
               inputs: Sequence[str] = [], 
               outputs: Sequence[str] = []) -> str:
    """Preprocess the input file using the script."""
    return f"{script} {input_file} {output_file} > {stdout} 2> {stderr}"

# Create preprocess output relative to the input directory for the workflow
def preprocess_config(
        input_dir = None, 
        output_dir = None, 
        model = "", 
        dataset = "", 
        model_dir = None,
        ) :
    """Create dataset specifc config."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir) if model_dir else None

    output_dir = output_dir / model / dataset

    if not model_dir:
        script = "preprocess.sh"
    else:
        script = model_dir / "preprocess.sh"

    # Check if the script is in search path
    if shutil.which(script):
        logger.debug(f"Found script {script} and it is in search path.")

    elif os.path.isfile(script):
        logger.debug(f"Found script {script}.")
        # Make absolute path
        script = os.path.abspath(script)

        # if bash script add bash command
        if  script.endswith("sh"):
            script = f"sh {script}"
        elif script.endswith("py"):
            script = f"python {script}"
        else:
            logger.error(f"Script {script} is not a bash or python script.")
            raise FileNotFoundError(f"Script {script} is not a bash or python script.")
    else:
        raise FileNotFoundError(f"Script {script} does not exist.")
    
    logger.debug(f"Preprocessing script: {script}")

    return (input_dir, output_dir , script )



def workflow(config: csa.Config,
             model_config: dict = None, 
             input_dir: str = None , 
             output_dir: str = None,
             model_name: str = None,):

    models = None

    print(f"config: {config}")

    # Check if the model is specified in the configuration file and assign it to the models variable
    if "model" not in model_config:
        logger.warning("Model not specified in the hyperparameter configuration file.")
        models = model_config # This is a dictionary of models
    else:
        models = {model_config["model"]}
  
    # Iterate over the models
    for model in models:
        if model == model_name or model_name == "all":
            for dataset in models[model]:
                logger.info(f"Preprocessing dataset {dataset} for {model}")
                (pp_input_dir, pp_output_dir , script) = preprocess_config(input_dir, 
                                                                           output_dir, 
                                                                           model, dataset, 
                                                                           model_dir=config.model_dir)
                preprocess(input_dir = pp_input_dir,
                           output_dir = pp_output_dir,
                           script = script,
                           env = config.conda_env,)
        else:
            logger.debug(f"Skipping model {model}")
            continue

    logger.info("Workflow completed.")
    return

def main(config: csa.Config):
    """Main function for the preprocessing workflow."""
    logger.info("Starting preprocessing workflow.")
    model_config = config.models_params


    workflow(config=config,
             model_config=config.models_params, 
             model_name=config.model_name,
             input_dir=config.input_dir,
             output_dir=config.output_dir,
            )   
            
    
    logger.info("Preprocessing workflow completed.")


if __name__ == "__main__":
    # Initialize the CLI
    config = csa.Config()
    
    params = config.initialize_parameters()
    logger.setLevel(config.log_level)

    config_params = config.params
    logger.info("Configuration parameters:")
    logger.debug(f"Config params: {config_params}")

    # Run the main function
    main(config)
    # workflow(config_params)

    sys.exit(0)

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


import csa_params_def as CSA
import improvelib.utils as frm
import improvelib.config.csa as csa

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))


@bash_app
def preprocess(input_file: str, output_file: str, script: str, stdout: str, stderr: str):
    """Preprocess the input file using the script."""
    return f"{script} {input_file} {output_file} > {stdout} 2> {stderr}"


def workflow(config: dict):

    models = None

    # Check if the model is specified in the configuration file and assign it to the models variable
    if "model" not in config:
        logger.warning("Model not specified in the hyperparameter configuration file.")
        models = config # This is a dictionary of models
    else:
        models = {config["model"]}
  
    # Iterate over the models
    for model in models:
        if model == "all":
            for dataset in models[model]:
                logger.info(f"Preprocessing dataset {dataset} for {model}")
                preprocess_dataset(dataset, config)
        else:
            logger.debug(f"Skipping model {model}")
            continue
    return

if __name__ == "__main__":
    # Initialize the CLI
    config = csa.Config()
    
    params = config.initialize_parameters()
    logger.setLevel(config.log_level)

    config_params = config.params
    logger.info("Configuration parameters:")
    logger.debug(f"Config params: {config_params}")


    # workflow(config_params)
    logger.info("Workflow completed.")
    sys.exit(0)

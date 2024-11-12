import sys
import json
import os
import time
#from time import time
from typing import Sequence, Tuple, Union
from pathlib import Path
import logging
import shutil

import parsl
from parsl import bash_app
from parsl.app.app import python_app
import parsl.concurrent
from parsl.config import Config
import parsl.config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.data_provider.files import File
# from parsl.data_provider.staging import Staging




import csa_params_def as CSA
import improvelib.utils as frm
import improvelib.config.csa as csa

from common import make_path, make_call
from apps import train, infer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))


# Create training output relative to the input directory for the workflow
def train_config(
        input_dir = None, 
        output_dir = None, 
        model = "", 
        dataset = "",
        target_dataset = "",
        source_dataset = "",
        split = "", 
        model_dir = None,
        ) :
    """Create dataset specifc config."""

    # Output dir is under output_dir/stage/model/source-target-dataset/split

    if not input_dir:
        raise FileNotFoundError("Input directory is not specified.")
    if not output_dir:
        output_dir = input_dir

    # constant for the stage
    stage = "train"

    # Check if the source and target datasets are specified
    if not source_dataset or not target_dataset:
        raise ValueError("Source and target datasets are not specified.")

    # Create directory paths
    train_input_dir = make_path(input_dir, "preprocess", model, source_dataset, target_dataset, split)
    train_output_dir = os.path.join( output_dir , stage , model , source_dataset , split)

    # should come from future / output of preprocess
    train_input_dir = os.path.join( input_dir , "preprocess" , model , "-".join([source_dataset, target_dataset]) , split)

    # Create the output directory
    if not os.path.exists(train_output_dir):
        logger.debug(f"Creating output directory: {train_output_dir}")
        os.makedirs(train_output_dir, exist_ok=True)
    if not os.path.exists(train_input_dir):
        logger.error(f"Missing input directory: {train_input_dir}")
        raise FileNotFoundError(f"Missing input directory: {train_input_dir}")
    

    # Train input  data
    inputs = { 
        "files": {
            },
        "input_dir" : train_input_dir,
        "output_dir" : train_output_dir,
        "stdout" : os.path.join(train_output_dir , "stdout.txt"),
        "stderr" : os.path.join(train_output_dir , "stderr.txt")
        }
       

    return inputs

def infer_config(
        input_dir = None, 
        output_dir = None, 
        model = "", 
        dataset = "",
        target_dataset = "",
        source_dataset = "",
        split = "", 
        model_dir = None,
        ) :
    """Create dataset specifc config."""

    # Output dir is under output_dir/stage/model/source-target-dataset/split

    if not input_dir:
        raise FileNotFoundError("Input directory is not specified.")
    if not output_dir:
        output_dir = input_dir

    # constant for the stage
    stage = "infer"

    # Check if the source and target datasets are specified
    if not source_dataset or not target_dataset:
        raise ValueError("Source and target datasets are not specified.")

    # Create directory paths
    step_output_dir = make_path(base_dir=output_dir, stage=stage, model=model, source_dataset=source_dataset, target_dataset=target_dataset, split=split)
    step_input_dir = make_path(base_dir=input_dir, stage="preprocess", model=model, source_dataset=source_dataset, target_dataset=target_dataset, split=split, make_dir=False)

    # should come from future / output of training
    trained_model_dir = None
    try: 
        trained_model_dir = make_path(base_dir=input_dir, stage="train", model=model, source_dataset=source_dataset, target_dataset=target_dataset, split=split, make_dir=False)
    except FileNotFoundError as e:
        logger.warning(f"Model directory not found for {model} {source_dataset} {target_dataset} {split}")
        try:
            trained_model_dir = make_path(base_dir=output_dir, stage="train", model=model, source_dataset=source_dataset, target_dataset=None, split=split, make_dir=False)
            logger.debug(f"Found model directory for {model} {source_dataset} {split}")
        except FileNotFoundError as e:
            logger.error(f"Model directory not found for {model} {source_dataset} {target_dataset} {split}")
            logger.error(f"Model directory not found for {model} {source_dataset} {split}")
            raise e
    
    if not trained_model_dir:
        raise FileNotFoundError("Model directory is not specified.")
    

    

    # Train input  data
    inputs = { 
        "files": {
            },
        "input_dir" : step_input_dir,
        "output_dir" : step_output_dir,
        "model_dir" : trained_model_dir,
        "stdout" : os.path.join(step_output_dir , "stdout.txt"),
        "stderr" : os.path.join(step_output_dir , "stderr.txt")
        }
       

    return inputs

def _check_executable(script: str = None, 
                      model_dir: str = None , 
                      model_name: str = None , 
                      conda_env: str = None):
    """Check if the script is an executable."""
    # Check if script is in the model directory or search path
    model_dir = Path(model_dir) if model_dir else None

    if not model_dir:
        script = Path(script)
    else:
        model_dir = Path(model_dir)
        script = model_dir / script

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

    logger.debug(f"Training script: {script}")
    return script

def workflow(config: csa.Config,
             model_config: dict = None, 
             input_dir: str = None , 
             output_dir: str = None,
             model_name: str = None,):

    models = None

    

    print(f"config: {config}")

    # Check if the model is specified in the configuration file and assign it to the models variable
    if "model" not in model_config:
        logger.warning(f"Model not specified in the hyperparameters configuration file: {config.cli.args.hyperparameters_file}.")
        models = model_config # This is a dictionary of models
    else:
        models = {model_config["model"]}

    if "splits" not in config.__dict__:
        raise ValueError("Splits are not specified in the configuration file.")
    
    train_script = _check_executable(script="train.sh" ,model_dir = config.model_dir, model_name = model_name)
    infer_script = _check_executable(script="infer.sh", model_dir = config.model_dir, model_name = model_name)
    # Iterate over the models

    preprocess_futures = []
    train_futures = []
    infer_futures = []

    for model in models:
        logger.debug(f"Checking model {model}")
        if model == model_name or model_name == "all":
            logger.debug(f"Training for model {model}")
            for source in config.source_datasets:
                logger.info(f"Training dataset {source} for {model}")
                # need only one target dataset for trainig for now ; train file is source dataset specific and identical for all target datasets
                target = config.target_datasets[0]
                for split in config.splits:
                    # print(output_dir)
                    # print(os.getcwd())
                    logger.info(f"Trainig {model} on dataset {source} and {split}")

                    if model in model_config and source in model_config[model]:
                        logger.debug(f"Model {model} and source {source} found in the configuration.")
                    else:
                        logger.error(f"Model {model} and source {source} not found in the configuration.")
                        raise ValueError(f"Model {model} and source {source} not found in the configuration.")

                    options = train_config(
                        input_dir=input_dir, # input_dir is the output of the preprocess
                        output_dir=output_dir, 
                        model=model,
                        source_dataset=source, 
                        target_dataset=target,
                        split=split)
                    
                    logger.debug(f"Training with {train_script} for {source} and {split}")
                    future = train(
                        input_dir = options["input_dir"],
                        output_dir = options["output_dir"],
                        epochs = config.epochs,
                        learning_rate = model_config[model][source]['learning_rate'] if 'learning_rate' in model_config[model][source] else None,
                        batch_size = model_config[model][source]['batch_size'] if 'batch_size' in model_config[model][source] else None,
                        column_name = config.y_col_name,
                        script = train_script,
                        conda_env = config.conda_env,
                        inputs = [
                            File(options["input_dir"]),
                            ],
                        outputs = [
                            File(options["output_dir"]),
                            File(options["stderr"]),
                            File(options["stdout"]),
                        ],
                        stderr = options["stderr"],
                        stdout = options["stdout"],
                        )
                    train_futures.append(future)
               

        else:
            logger.debug(f"Skipping model {model}")
            continue

        
        while train_futures:

            logger.info(f"Waiting for training {len(train_futures)} tasks to complete.")
            for f in train_futures:

                # get source and split from future
                # future result is a dictionary with source and split
                
                if f.done():
                    if f.exception():
                        logger.error(f"Future(training) {f.tid} has an exception: {f.exception()}")
                    else:
                        logger.info(f"Future(training) {f.tid} is done.")
                        # logger.info(f"Output: {f.result()}")
                    
                    train_futures.remove(f)

                    model_dir = f.outputs[0].result().filepath
                    elements = model_dir.split("/")
                    split = elements[-1]
                    source = elements[-2]
                    
                    logger.info(f"Training task {f.tid} completed: {model} {source} {split}")
                    logger.debug(f"Path to model: {model_dir}")
                    for target in config.target_datasets:
                        logger.info(f"Infering {model} on dataset {source} and {target} for split {split}")
                        # infer(source, target, split)
                        options = infer_config(
                            input_dir=input_dir, # input_dir is the output of the preprocess
                            output_dir=output_dir, 
                            model=model,
                            source_dataset=source, 
                            target_dataset=target,
                            split=split)
                        
                        i_future = infer(
                                    script = infer_script,
                                    input_data_dir = options["input_dir"],
                                    input_model_dir = model_dir, # infer_options["model_dir"],
                                    output_dir = options["output_dir"],
                                    calc_infer_scores = True,
                                    y_col_name = config.y_col_name,
                                    conda_env = config.conda_env,
                                    inputs = [
                                        File(options["input_dir"]),
                                        File(options["model_dir"]),
                                    ],
                                    outputs = [
                                        File(options["output_dir"]),  
                                        File(options["stdout"]),
                                        File(options["stderr"]),
                                    ],
                                    stderr = options["stderr"],
                                    stdout = options["stdout"],
                                    )
                        logger.debug(f"Inference task {i_future.tid} submitted: {model} {source} {target} {split}")
                        infer_futures.append(i_future)

            # Wait for all the futures to complete
            time.sleep(30)
            
        logger.info("Waiting for infer tasks to complete.")

    while infer_futures:

        # Check if the future is done
        for future in infer_futures:
            if future.done():
                # Check if the future has an exception
                if future.exception():
                    logger.error(f"Future {future} has an exception: {future.exception()}")
                else:
                    logger.info(f"Future {future} is done.")
                    logger.info(f"Output: {future.result()}")

                # remove the future from the list
                infer_futures.remove(future)
            else:
                logger.info(f"Future {future} is not done.")

        # sleep for 5 seconds
        time.sleep(30)

    logger.info("Infering tasks completed.")



    logger.info("Workflow completed.")
    return


def init_parsl(config: parsl.config.Config):
    """Initialize the Parsl configuration."""

    # Check if the Parsl configuration is specified
    if config is None:
        logger.error("Parsl configuration is not specified.")
        raise ValueError("Parsl configuration is not specified.")
    
    
    # Disable logging for Parsl
    config.initialize_logging = False
    config.usage_tracking = True
    
    # Load the Parsl configuration
    logger.info("Initializing Parsl configuration.")
    parsl.clear()
    parsl.load(config)
    logger.info("Parsl configuration initialized.")
    return

def shutdown_parsl():
    """Shutdown the Parsl configuration."""
    logger.info("Shutting down Parsl configuration.")
    parsl.dfk().cleanup()
    parsl.clear()
    logger.info("Parsl configuration shutdown.")
    return

def main(config: csa.Config):
    """Main function for the training workflow."""
    logger.info("Starting training workflow.")
    model_config = config.model_params


    print(model_config)
   
    init_parsl(config.parsl_config)
    results = workflow(config=config,
                        model_config=config.model_params, 
                        model_name=config.model_name,
                        input_dir=config.input_dir,
                        output_dir=config.output_dir,
                        )
    # print(config.parsl_config.get_usage_information())
    shutdown_parsl()   
            
    
    logger.info("Training workflow completed.")


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

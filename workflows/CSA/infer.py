import sys
import json
import os
from time import time
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
import time


from common import make_call
from common import make_path 



import csa_params_def as CSA
import improvelib.utils as frm
import improvelib.config.csa as csa

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))


@bash_app
def preprocess( 
    input_dir = None,
    output_dir = None,
    train_file = None, 
    val_file = None,
    test_file = None,
    column_name = None,
    conda_env = None,
    script = None,
    
    stderr = "stderr.txt",
    stdout = "stdout.txt",
    inputs = [],
    outputs = [], # File(os.path.join(os.getcwd(), 'my_stdout*'))
    ):
    """Preprocess the input file using the script."""
    

    # Prefix and activate the conda environment
    if conda_env:
        script = f"START=$(date +%s) ; echo Start:\t$START; conda_path=$(dirname $(dirname $(which conda))); source $conda_path/bin/activate {conda_env} ; {script}"

    # Create the command line interface for preprocessing
    cli = [ "time",
           script,
        "--train_split_file" , train_file,
        "--val_split_file" , val_file,
        "--test_split_file" , test_file,
        "--input_dir" , input_dir,
        "--output_dir" , output_dir,
        "--y_col_name" , column_name
    ]

    call = " ".join(cli)

    SUFFIX=' ; STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1'
    call = call + SUFFIX

    logger.debug(f"Preprocessing command: {call}")
    return call
    # return f"cd {output_dir} ; {script} {input_dir} {output_dir} | tee my_stdout.txt"



def _check_executable(script: str = None, 
                      model_dir: str = None , 
                      model_name: str = None , 
                      conda_env: str = None):
    """Check if the script is an executable."""
    # Check if script is in the model directory or search path
    model_dir = Path(model_dir) if model_dir else None

    if not model_dir:
        script = "infer.sh"
    else:
        model_dir = Path(model_dir)
        script = model_dir / "infer.sh"

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

    logger.debug(f"Infer script: {script}")
    return script

@bash_app
def infer(
    script: str = None,
    input_data_dir: str = None,
    input_model_dir: str = None,
    output_dir: str = None,
    calc_infer_scores: bool = False,
    y_col_name: str = None,
    conda_env: str = None,
    stdout: str = "stdout.txt",
    stderr: str = "stderr.txt",
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = []):
    """Infer the model."""
    call = "echo 'Inferencing the model'"
    prefix = f"START=$(date +%s) ; echo Start:\t$START "

    import logging
    logger = logger = logging.getLogger(__name__)
    
    if conda_env:
        conda= f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {conda_env} "
    else:
        conda = "echo no conda env provided"

    suffix = "STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1"

   
    # Create the command line interface for inference
    cli = [ "time",
              script,
          "--input_data_dir" , input_data_dir,
          "--input_model_dir" , input_model_dir,
          "--output_dir" , output_dir,
          "--calc_infer_scores" , calc_infer_scores,
          "--y_col_name" , y_col_name
     ]

    # create a list of strings from cli
    call = " ;".join([prefix, conda, " ".join([ str(i) for i in cli]), suffix])

    logger.debug(f"Inference command: {call}")
    print(call)
    return call


# Create training output relative to the input directory for the workflow
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
        try:
            trained_model_dir = make_path(base_dir=input_dir, stage="train", model=model, source_dataset=source_dataset, target_dataset=None, split=split, make_dir=False)
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

def workflow(config: csa.Config,
             model_config: dict = None, 
             input_dir: str = None , 
             output_dir: str = None,
             model_name: str = None,):

    models = None

    

    logger.info(f"config: {config}")

    # Check if the model is specified in the configuration file and assign it to the models variable
    if "model" not in model_config:
        logger.warning("Model not specified in the hyperparameter configuration file.")
        models = model_config # This is a dictionary of models
    else:
        models = {model_config["model"]}

    if "splits" not in config.__dict__:
        raise ValueError("Splits are not specified in the configuration file.")
    
    script = _check_executable(model_dir = config.model_dir, model_name = model_name)
  
    # Iterate over the models
    infer_futures = []

    for model in models:
        logger.debug(f"Checking model {model}")
        if model == model_name or model_name == "all":
            logger.debug(f"Infer for model {model}")
            for source in config.source_datasets:
                for target in config.target_datasets:
                    for split in config.splits:
                        logger.info(f"Infer {model} on dataset {source}-{target} and {split}")

                        if model in model_config and source in model_config[model]:
                            logger.debug(f"Model {model} and source {source} found in the configuration.")
                        else:
                            logger.error(f"Model {model} and source {source} not found in the configuration.")
                            raise ValueError(f"Model {model} and source {source} not found in the configuration.")

                        infer_options = infer_config(
                            input_dir=output_dir, # input_dir is the output of the preprocess
                            output_dir=output_dir, 
                            model=model,
                            source_dataset=source, 
                            target_dataset=target,
                            split=split)
                        
                        logger.debug(f"Infer options: {infer_options}")
                        future = infer(
                            script = script,
                            input_data_dir = infer_options["input_dir"],
                            input_model_dir = infer_options["model_dir"],
                            output_dir = infer_options["output_dir"],
                            calc_infer_scores = True,
                            y_col_name = config.y_col_name,
                            conda_env = config.conda_env,
                            inputs = [
                                File(infer_options["input_dir"]),
                                File(infer_options["model_dir"]),
                            ],
                            outputs = [
                                File(infer_options["output_dir"]),  
                                File(infer_options["stdout"]),
                                File(infer_options["stderr"]),
                            ],
                            stderr = infer_options["stderr"],
                            stdout = infer_options["stdout"],
                            )
                        logger.debug(f"Inference task {future.tid} submitted: {model} {source} {target} {split}")
                        infer_futures.append(future)
                        
        else:
            logger.debug(f"Skipping model {model}")
            continue

    while infer_futures :
        for future in infer_futures:
            if future.done():
                logger.info(f"Future(infer) {future.tid} is done.")
                # remove the future from the list
                infer_futures.remove(future)

                for data in future.outputs:
                    if data.done():
                        # print(data.result().url)
                        # print(data.filepath)
                        # print(data.file_obj)
                        if os.path.isfile(data.filepath):
                            print(f"{data.tid} is done.")
                            print(f"Name {data.filename} is a file.")
                        elif os.path.isdir(data.filepath):
                            print(f"{data.tid} is done.")
                            print(f"Name {data.filename} is a directory.")
                        else:
                            print(f"Data {data.tid} is neither file nor directory.")
                    else:
                        print(f"Data {data.tid} - {data.filename}  is not done.")
        logger.info(f"Waiting for 30 seconds. {len(infer_futures)} infer tasks remaining.")    
        time.sleep(30)    


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
    """Main function for the preprocessing workflow."""
    logger.info("Starting preprocessing workflow.")
    model_config = config.model_params

    init_parsl(config.parsl_config)
    results = workflow(config=config,
                        model_config=config.model_params, 
                        model_name=config.model_name,
                        input_dir=config.input_dir,
                        output_dir=config.output_dir,
                        )
    # print(config.parsl_config.get_usage_information())
    shutdown_parsl()   
            
    
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

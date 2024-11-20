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
# import improvelib.utils as frm
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
    
    import logging 
    logger = logging.getLogger(__name__)

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

# Create preprocess output relative to the input directory for the workflow
def preprocess_config(
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
    stage = "preprocess"

    # Check if the source and target datasets are specified
    if not source_dataset or not target_dataset:
        raise ValueError("Source and target datasets are not specified.")

    # Create directory paths
    pp_output_dir = os.path.join( output_dir , stage , model , "-".join([source_dataset, target_dataset]) , split)
    pp_input_dir = os.path.join( input_dir ) 

    # Create the output directory
    if not os.path.exists(pp_output_dir):
        logger.debug(f"Creating output directory: {pp_output_dir}")
        os.makedirs(pp_output_dir, exist_ok=True)
    if not os.path.exists(pp_input_dir):
        logger.debug(f"Creating input directory: {pp_input_dir}")
        os.makedirs(pp_input_dir, exist_ok=True)

    if source_dataset == target_dataset:
            # If source and target are the same, then infer on the test split
            test_split_file = f"{source_dataset}_split_{split}_test.txt"
    else:
            # If source and target are different, then infer on the entire target dataset
            test_split_file = f"{target_dataset}_all.txt"

    # Preprocess  data
    inputs = { 
        "files": {
            "train" : f"{source_dataset}_split_{split}_train.txt",
            "val" : f"{source_dataset}_split_{split}_val.txt",
            "test" : test_split_file},
        "input_dir" : pp_input_dir,
        "output_dir" : pp_output_dir,
        "stdout" : os.path.join(pp_output_dir , "stdout.txt"),
        "stderr" : os.path.join(pp_output_dir , "stderr.txt")
        }
       

    return (inputs, pp_input_dir, pp_output_dir , os.path.join( pp_output_dir ,  "stderr.txt")  , os.path.join( pp_output_dir , "stdout.txt"))

@bash_app
def train( 
    script: str = None, 
    input_dir: str = None, 
    output_dir: str = None, 
    epochs: int = 1, 
    column_name: str = None, 
    learning_rate: float = 0.001, 
    batch_size: int = 32 , 
    conda_env: str = None,
    stdout: str = "stdout.txt", 
    stderr: str = "stderr.txt", 
    inputs: Sequence[File] = [], 
    outputs: Sequence[File] = [] ):    
    """Train the model."""
    call= "echo 'Training the model'"

    prefix = f"START=$(date +%s) ; echo Start:\t$START ; "

    if conda_env:
        conda= f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {conda_env} ; "
    else:
        conda = ""

    suffix = "STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1"

    cli = [ script, 
           "--input_dir" , input_dir,
           "--output_dir" , output_dir,
           "--epochs" , epochs,
           "--y_col_name" , column_name,
           "--learning_rate" , learning_rate,
           "--batch_size" , batch_size
           ]

    call = echo + " ".join(cli)

    return call

def _check_executable(script: str = None, 
                      model_dir: str = None , 
                      model_name: str = None , 
                      conda_env: str = None):
    """Check if the script is an executable."""
    # Check if script is in the model directory or search path
    model_dir = Path(model_dir) if model_dir else None

    if not model_dir:
        script = "preprocess.sh"
    else:
        model_dir = Path(model_dir)
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
        logger.warning("Model not specified in the hyperparameter configuration file.")
        models = model_config # This is a dictionary of models
    else:
        models = {model_config["model"]}

    if "splits" not in config.__dict__:
        raise ValueError("Splits are not specified in the configuration file.")
    
    script = _check_executable(model_dir = config.model_dir, model_name = model_name)
  
    # Iterate over the models

    preprocess_futures = []

    for model in models:
        logger.debug(f"Checking model {model}")
        if model == model_name or model_name == "all":
            logger.debug(f"Preprocessing for model {model}")
            for source in config.source_datasets:
                logger.info(f"Preprocessing dataset {source} for {model}")
                for target in config.target_datasets:
                    for split in config.splits:
                        print(output_dir)
                        print(os.getcwd())
                        logger.info(f"Preprocessing dataset {source} for {model} and {split}")
                        (options, pp_input_dir, pp_output_dir , stdout_file , stderr_file ) = preprocess_config(input_dir=input_dir,
                                                                           output_dir=output_dir, 
                                                                           model=model,
                                                                           source_dataset=source, 
                                                                           target_dataset=target,
                                                                           split=split)
                        logger.debug(f"Preprocessing with {script} for {source} and {target} in {split}")
                        future = preprocess(
                            input_dir = options["input_dir"],
                            output_dir = options["output_dir"],
                            train_file = options["files"]["train"],
                            val_file = options["files"]["val"],
                            test_file = options["files"]["test"],
                            column_name = config.y_col_name,
                            script = script,
                            conda_env = config.conda_env,
                            inputs = [
                                File(options["input_dir"]),
                                File("/".join([options["input_dir"], "splits" , options["files"]["train"]])),
                                File("/".join([options["input_dir"], "splits" , options["files"]["val"]])),
                                File("/".join([options["input_dir"], "splits" , options["files"]["test"]])),],
                            outputs = [
                                File(options["output_dir"]),
                                File(os.path.join(options["output_dir"], "stderr.txt")),
                                File(os.path.join(options["output_dir"], "stdout.txt"))
                            ],
                            stderr = os.path.join(options["output_dir"], "stderr.txt"),
                            stdout = os.path.join(options["output_dir"], "stdout.txt"),
                            )
                        logger.debug(f"Preprocessing task {future.tid} submitted: {model} {source} {target} {split}")
                        preprocess_futures.append(future)
                       
        else:
            logger.debug(f"Skipping model {model}")
            continue

    # Wait for all the futures to complete
    logger.info("Waiting for all the preprocessing tasks to complete.")
    for future in preprocess_futures:
        # print(future.__dict__)
        print(future.outputs)
        print(future.stderr)
        print(future.result())

    while preprocess_futures:
        for future in preprocess_futures:
            if future.done():
                print(f"Future {future.tid} is done.")
                # remove the future from the list
                preprocess_futures.remove(future)
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
                        print(f"Data {data.tid} is not done.")
                # sleep for 10 seconds
            time.sleep(10)    
        


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

import sys
import os
import time
from pathlib import Path
import logging
import shutil
import importlib

import parsl
from parsl import bash_app
#from parsl.app.app import python_app
#import parsl.concurrent
from parsl.config import Config
import parsl.config
#from parsl.executors import HighThroughputExecutor
#from parsl.providers import LocalProvider
from parsl.data_provider.files import File
# from parsl.data_provider.staging import Staging


#import csa_params_def as CSA
#import improvelib.config.csa as csa
from improvelib.initializer.config import Config
import csa_parsl_params_def

filepath = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))


@bash_app
def preprocess(script_call = None, conda_env = None, stderr = "stderr.txt", stdout = "stdout.txt", inputs = [], outputs = []):
    """Preprocess the input file using the script."""
    import logging 
    logger = logging.getLogger(__name__)
    # Prefix and activate the conda environment
    prefix = f"START=$(date +%s) ; echo Start:\t$START; conda_path=$(dirname $(dirname $(which conda))); source $conda_path/bin/activate {conda_env} ; {script_call}"
    SUFFIX=' ; STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1'
    call = prefix + script_call + SUFFIX
    logger.debug(f"Preprocessing command: {call}")
    return call




def _check_executable(script: str = None, model_dir: str = None , model_name: str = None ):
    """Check if the model script is valid."""
    # Create model script name
    model_preprocess_path = Path(model_dir) / f"{model_name}_preprocess_improve.py"
    # Check if model script exists
    if os.path.isfile(model_preprocess_path):
        logger.debug(f"Found script {model_preprocess_path}.")
        # Make absolute path
        script = os.path.abspath(model_preprocess_path)
    else:
        raise FileNotFoundError(f"Script {script} does not exist.")
    logger.debug(f"Preprocessing script: {script}")
    return script

def workflow(params):

    #models = None
    #print(f"config: {config}")
    # Check if the model is specified in the configuration file and assign it to the models variable
    #if "model" not in model_config:
    #    logger.warning("Model not specified in the hyperparameter configuration file.")
    #    models = model_config # This is a dictionary of models
    #else:
    #    models = {model_config["model"]}
    #if "splits" not in config.__dict__:
    #    raise ValueError("Splits are not specified in the configuration file.")
    
    script = _check_executable(model_dir = params['model_scripts_dir'], model_name = params['model_name'])
    preprocess_futures = []

    # Iterate over the datasets
    logger.debug(f"Preprocessing for model {params['model_name']}")
    for source in params['source_datasets']:
        logger.info(f"Preprocessing dataset {source} for {params['model_name']}")
        for target in params['target_datasets']:
            for split in params['split']:
                #print(output_dir)
                #print(os.getcwd())
                logger.info(f"Preprocessing dataset {source} for {params['model_name']} and {split}")
                # Create directory paths
                ml_data_dir = os.path.join(params['output_dir'] , "preprocess" , params['model_name'], "-".join([source, target]) , split)
                input_dir = Path(params['input_dir'])
                if source == target:
                    # If source and target are the same, then infer on the test split
                    test_split_file = f"{source}_split_{split}_test.txt"
                else:
                    # If source and target are different, then infer on the entire target dataset
                    test_split_file = f"{target}_all.txt"
                train_split_file = f"{source}_split_{split}_train.txt"
                val_split_file = f"{source}_split_{split}_val.txt"
                logger.debug(f"Preprocessing with {script} for {source} and {target} in {split}")
                    # Create the command line interface for preprocessing
                script_call = [ "time",
                        str(script),
                        "--train_split_file" , str(train_split_file),
                        "--val_split_file" , str(val_split_file),
                        "--test_split_file" , str(test_split_file),
                        "--input_dir" , str(input_dir),
                        "--output_dir" , str(ml_data_dir)]
                script_call = " ".join(script_call)
                future = preprocess(script_call = script_call,
                                    conda_env = params['model_environment'],
                                    inputs = [
                                        File(params["input_dir"]),
                                        File("/".join([input_dir, "splits" , train_split_file])),
                                        File("/".join([input_dir, "splits" , val_split_file])),
                                        File("/".join([input_dir, "splits" , test_split_file])),
                                        ],
                                    outputs = [
                                        File(ml_data_dir),
                                        File(os.path.join(ml_data_dir, "stderr.txt")),
                                        File(os.path.join(ml_data_dir, "stdout.txt"))
                                        ],
                                    stderr = os.path.join(ml_data_dir, "stderr.txt"),
                                    stdout = os.path.join(ml_data_dir, "stdout.txt"),
                                    )
                logger.debug(f"Preprocessing task {future.tid} submitted: {params['model_name']} {source} {target} {split}")
                preprocess_futures.append(future)
                       
        else:
            logger.debug(f"Skipping model {params['model_name']}")
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

def _load_parsl_config(parsl_config_file, available_accelerators):
    # if the config file is a yaml file load it as yaml
    if parsl_config_file.endswith('.yaml'):
        logger.error("YAML format is not supported for parsl_config_file")
        sys.exit(1)
        with open(self.parsl_config_file, 'r') as f:
            return yaml.safe_load(f)
        
    # if the config file is a json file load it as json
    elif parsl_config_file.endswith('.json'):
        logger.error("JSON format is not supported for parsl_config_file")
        sys.exit(1)
        with open(self.parsl_config_file, 'r') as f:
            return json.load(f)
        
    # if the config file is python file import it and assign the importet parsl_config to self.parsl_config
    elif parsl_config_file.endswith('.py'):
        # import the parsl_config from the file
        # add the directory of the config file to the path
        sys.path.append(os.path.dirname(parsl_config_file))
        # get filename without extension
        filename = os.path.basename(parsl_config_file).split('.')[0]
        # import the file
        imported_parsl_config = importlib.import_module(filename)
        # assign the parsl_config to self.parsl_config
        parsl_config = imported_parsl_config.get_parsl_config(available_accelerators)
    else:
        logger.error("Unknown file format for parsl_config_file")
        sys.exit(1)
    return parsl_config

def init_parsl(parsl_config_file):
    """Initialize the Parsl configuration."""
    parsl_config = _load_parsl_config(parsl_config_file, params['available_accelerators'])
    
    # Disable logging for Parsl
    parsl_config.initialize_logging = False
    parsl_config.usage_tracking = True
    
    # Load the Parsl configuration
    logger.info("Initializing Parsl configuration.")
    parsl.clear()
    parsl.load(parsl_config)
    logger.info("Parsl configuration initialized.")
    return

def shutdown_parsl():
    """Shutdown the Parsl configuration."""
    logger.info("Shutting down Parsl configuration.")
    parsl.dfk().cleanup()
    parsl.clear()
    logger.info("Parsl configuration shutdown.")
    return

def main(params):
    """Main function for the preprocessing workflow."""
    logger.info("Starting preprocessing workflow.")
    #model_config = config.model_params
    #init_parsl(config.parsl_config)
    init_parsl(params['parsl_config_file'])
    results = workflow(params)
    shutdown_parsl()   
    logger.info("Preprocessing workflow completed.")


if __name__ == "__main__":
    # Initialize the CLI
    cfg = Config() 
    params = cfg.initialize_parameters(
        section="CSA",
        pathToModelDir=filepath,
        default_config="csa_parsl_params.ini",
        additional_definitions=csa_parsl_params_def.additional_definitions)
    
    #config = csa.Config()
    #params = config.initialize_parameters()
    #logger.setLevel(config.log_level)

    #config_params = config.params
    logger.info("Configuration parameters:")
    #logger.debug(f"Config params: {config_params}")

    # Run the main function
    #main(config)
    main(params)
    sys.exit(0)
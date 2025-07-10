import sys
import os
import time
from pathlib import Path
import logging

import parsl
from parsl.data_provider.files import File

from improvelib.initializer.config import Config
from workflows.utils_workflows import check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_string
from workflows.utils_parsl import init_parsl, shutdown_parsl, check_model_script, make_call
import csa_parsl_params_def

filepath = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))




def workflow(params):
    print(params)
    model_env = check_dir_path_or_model_scripts_dir(params['model_environment'], params['model_scripts_dir'])
    script = check_model_script(model_dir = params['model_scripts_dir'], model_name = params['model_name'], stage = "preprocess")

    # Prepare additional parameters
    preprocess_additional_args = get_additional_parameters(params['preprocess_args'])
    if 'input_supp_data_dir' in preprocess_additional_args:
        preprocess_additional_args['input_supp_data_dir'] = check_dir_path_or_model_scripts_dir(preprocess_additional_args['input_supp_data_dir'], params['model_scripts_dir'])
    preprocess_additional_args = additional_parameters_dict_to_string(preprocess_additional_args)

    preprocess_futures = []

    # Iterate over the datasets
    logger.debug(f"Preprocessing for model {params['model_name']}")
    for source in params['source_datasets']:
        logger.info(f"Preprocessing dataset {source} for {params['model_name']}")
        for target in params['target_datasets']:
            for split in params['split_nums']:
                logger.info(f"Preprocessing dataset {source} for {params['model_name']} and {split}")
                # Create directory paths
                ml_data_dir = os.path.join(params['output_dir'] , "ml_data" ,  f"{source}-{target}" , f"split_{split}")
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
                script_call = [ "python",
                        str(script),
                        "--train_split_file" , str(train_split_file),
                        "--val_split_file" , str(val_split_file),
                        "--test_split_file" , str(test_split_file),
                        "--input_dir" , str(input_dir),
                        "--output_dir" , str(ml_data_dir)]
                script_call = " ".join(script_call)
                script_call = script_call + preprocess_additional_args
                future = make_call(script_call = script_call,
                                    conda_env = model_env,
                                    inputs = [
                                        File(params["input_dir"]),
                                        File("/".join([str(input_dir), "splits" , train_split_file])),
                                        File("/".join([str(input_dir), "splits" , val_split_file])),
                                        File("/".join([str(input_dir), "splits" , test_split_file])),
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



def main(params):
    """Main function for the preprocessing workflow."""
    logger.info("Starting preprocessing workflow.")
    logger.info("Initializing Parsl configuration.")
    init_parsl(params['parsl_config_file'], params['available_accelerators'])
    logger.info("Parsl configuration initialized.")
    workflow(params)
    logger.info("Shutting down Parsl configuration.")
    shutdown_parsl()
    logger.info("Parsl configuration shutdown.")
    logger.info("Preprocessing workflow completed.")


if __name__ == "__main__":
    # Initialize the CLI
    cfg = Config() 
    print(Config)
    params = cfg.initialize_parameters(
        section="CSA",
        pathToModelDir=filepath,
        default_config="csa_parsl_params.ini",
        additional_definitions=csa_parsl_params_def.additional_definitions)
    
    print(params)
    # Run the main function
    main(params)
    sys.exit(0)
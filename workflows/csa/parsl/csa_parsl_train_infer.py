import sys
import os
import time
import logging
import parsl
from pathlib import Path
from parsl.data_provider.files import File

from improvelib.initializer.config import Config
from workflows.utils_workflows import check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_string
from workflows.utils_parsl import init_parsl, shutdown_parsl, check_model_script, make_call
import csa_parsl_params_def

filepath = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))



def workflow(params):
    model_env = check_dir_path_or_model_scripts_dir(params['model_environment'], params['model_scripts_dir'])
    train_script = check_model_script(model_dir = params['model_scripts_dir'], model_name = params['model_name'], stage = "train")
    infer_script = check_model_script(model_dir = params['model_scripts_dir'], model_name = params['model_name'], stage = "infer")

    train_additional_args = get_additional_parameters(params['train_args'])
    train_additional_args = additional_parameters_dict_to_string(train_additional_args)

    infer_additional_args = get_additional_parameters(params['infer_args'])
    infer_additional_args = additional_parameters_dict_to_string(infer_additional_args)

    preprocess_futures = []
    train_futures = []
    infer_futures = []


    for source in params['source_datasets']:
        logger.info(f"Training dataset {source}.")
        # need only one target dataset for trainig for now ; train file is source dataset specific and identical for all target datasets
        target = params['target_datasets'][0]
        for split in params['split_nums']:
            # print(output_dir)
            # print(os.getcwd())
            logger.info(f"Trainig on dataset {source} and {split}")
            # should come from future / output of preprocess
            train_input_dir = os.path.join(params['output_dir'], "ml_data", f"{source}-{target}", f"split_{split}")
            train_output_dir = os.path.join(params['output_dir'], "models", source, f"split_{split}")

            if not os.path.exists(train_input_dir):
                logger.error(f"Missing input directory: {train_input_dir}")
                raise FileNotFoundError(f"Missing input directory: {train_input_dir}")
            script_call = [ "python",
                            str(train_script),
                            "--input_dir" , str(train_input_dir),
                            "--output_dir" , str(train_output_dir)]
            script_call = " ".join(script_call)
            script_call = script_call + train_additional_args
            logger.debug(f"Training with {train_script} for {source} and {split}")
            future = make_call(
                script_call = script_call,
                conda_env = model_env,
                inputs = [File(train_input_dir)],
                outputs = [
                    File(train_output_dir),
                    File(os.path.join(train_output_dir, "stderr.txt")),
                    File(os.path.join(train_output_dir, "stdout.txt")),
                    File(os.path.join(train_output_dir, "val_scores.json"))
                    ],
                stderr = os.path.join(train_output_dir, "stderr.txt"),
                stdout = os.path.join(train_output_dir, "stdout.txt"),
                )
            train_futures.append(future)
               

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
        
                logger.info(f"Training task {f.tid} completed: {source} {split}")
                logger.debug(f"Path to model: {model_dir}")
                for target in params['target_datasets']:
                    logger.info(f"Infering on dataset {source} and {target} for split {split}")
                    infer_input_data_dir = os.path.join(params['output_dir'], "ml_data", f"{source}-{target}", split)
                    infer_input_model_dir = os.path.join(params['output_dir'], "models", source, split)
                    infer_output_dir = os.path.join(params['output_dir'], "infer", f"{source}-{target}", split)
                    if not os.path.exists(infer_input_model_dir):
                        logger.error(f"Missing input directory: {infer_input_data_dir}")
                        raise FileNotFoundError(f"Missing input directory: {infer_input_model_dir}")
                    script_call = [ "python",
                                    str(infer_script),
                                    "--input_data_dir" , str(infer_input_data_dir),
                                    "--input_model_dir" , str(infer_input_model_dir),
                                    "--output_dir" , str(infer_output_dir),
                                    "--calc_infer_scores true"]
                    script_call = " ".join(script_call)
                    script_call = script_call + infer_additional_args
                    i_future = make_call(
                                script_call = script_call,
                                conda_env = model_env,
                                inputs = [
                                    File(infer_input_data_dir),
                                    File(infer_input_model_dir),
                                ],
                                outputs = [
                                    File(infer_output_dir),  
                                    File(os.path.join(infer_output_dir, "stderr.txt")),
                                    File(os.path.join(infer_output_dir, "stdout.txt")),
                                    File(os.path.join(infer_output_dir, "test_scores.json"))
                                ],
                                stderr = os.path.join(infer_output_dir, "stderr.txt"),
                                stdout = os.path.join(infer_output_dir, "stdout.txt"),
                                )
                    logger.debug(f"Inference task {i_future.tid} submitted: {source} {target} {split}")
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




def main(params):
    """Main function for the training and inference workflow."""
    logger.info("Starting training and inference workflow.")
    logger.info("Initializing Parsl configuration.")
    init_parsl(params['parsl_config_file'], params['available_accelerators'])
    logger.info("Parsl configuration initialized.")
    workflow(params)
    logger.info("Shutting down Parsl configuration.")
    shutdown_parsl()
    logger.info("Parsl configuration shutdown.")
    logger.info("Training and inference workflow completed.")


if __name__ == "__main__":
    # Initialize the CLI
    cfg = Config() 
    params = cfg.initialize_parameters(
        section="CSA",
        pathToModelDir=filepath,
        default_config="csa_parsl_params.ini",
        additional_definitions=csa_parsl_params_def.additional_definitions)
    
    logger.info("Configuration parameters:")
    # Run the main function
    main(params)
    sys.exit(0)
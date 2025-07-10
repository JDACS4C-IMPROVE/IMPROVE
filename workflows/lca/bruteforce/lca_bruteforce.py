import os
import time
from pathlib import Path
import subprocess
import lca_bruteforce_params_def
from improvelib.initializer.config import Config
from workflows.utils_workflows import save_log, save_time, check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_list


# Sets up parameters
filepath = Path(__file__).resolve().parent
cfg = Config() 
params = cfg.initialize_parameters(
    section="LCA",
    pathToModelDir=filepath,
    default_config="lca_bruteforce_params.ini",
    additional_definitions=lca_bruteforce_params_def.additional_definitions
)

# Makes output_dir
output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)


# Creates names for model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Specify dirs
MAIN_ML_DATA_DIR = output_dir / 'ml_data' 
MAIN_MODEL_DIR = output_dir / 'models' 
MAIN_INFER_DIR = output_dir / 'infer' 
print("Created directory names.")
print("output_dir:  ", output_dir)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)


# Prepares additional parameters
preprocess_additional_args = get_additional_parameters(params['preprocess_args'])
if 'input_supp_data_dir' in preprocess_additional_args:
    preprocess_additional_args['input_supp_data_dir'] = check_dir_path_or_model_scripts_dir(preprocess_additional_args['input_supp_data_dir'], params['model_scripts_dir'])
preprocess_additional_args = additional_parameters_dict_to_list(preprocess_additional_args)

train_additional_args = get_additional_parameters(params['train_args'])
train_additional_args = additional_parameters_dict_to_list(train_additional_args)

infer_additional_args = get_additional_parameters(params['infer_args'])
infer_additional_args = additional_parameters_dict_to_list(infer_additional_args)

## Loops through all splits specified in the parameters
for split_num in params['split_nums']:

    split_name = "split_" + split_num
    print(f"Running LCA with {params['dataset']} {split_name}...")

    # Determines files for training shards from the lca_splits_dir
    lca_split_paths = list(Path(params['lca_splits_dir']).glob(f"{params['dataset']}_split_{split_num}_sz_*.txt"))
    lca_split_files = [os.path.basename(x) for x in lca_split_paths]
    print(f"Running LCA on {len(lca_split_files)} shards with the following training splits:", lca_split_files)

    # Sets val and test names
    val_split_file = f"{params['dataset']}_split_{split_num}_val.txt"
    test_split_file = f"{params['dataset']}_split_{split_num}_test.txt"

    ## Loops through all shards for the specified split
    for lca in lca_split_files:
        lca_train_path = params['lca_splits_dir'] + '/' + lca
        lca_name = "sz_" + lca.split('.')[0].split('_')[4]
        print(f"Running IMPROVE scripts with {lca} for training...")
        ### PREPROCESS
        ml_data_dir = MAIN_ML_DATA_DIR / split_name / lca_name
        preprocess_start = time.time()
        preprocess_run = ["python", preprocess_python_script,
                          "--train_split_file", str(lca_train_path),
                          "--val_split_file", str(val_split_file),
                          "--test_split_file", str(test_split_file),
                          "--input_dir", params['input_dir'], 
                          "--output_dir", str(ml_data_dir)] + preprocess_additional_args
        preprocess_result = subprocess.run(preprocess_run,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                
        # Log and Time
        print(f"preprocess returncode = {preprocess_result.returncode}")
        save_log(ml_data_dir, preprocess_result)
        save_time(ml_data_dir, preprocess_start)

        ### TRAIN
        train_start = time.time()
        model_dir = MAIN_MODEL_DIR / split_name / lca_name
        train_run = ["python", train_python_script,
                    "--input_dir", str(ml_data_dir),
                    "--output_dir", str(model_dir)] + train_additional_args
        train_result = subprocess.run(train_run,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
        # Log and Time
        print(f"train returncode = {train_result.returncode}")
        save_log(model_dir, train_result)
        save_time(model_dir, train_start)

        ### INFER
        infer_start = time.time()
        infer_dir = MAIN_INFER_DIR / split_name / lca_name
        infer_run = ["python", infer_python_script,
                     "--input_data_dir", str(ml_data_dir),
                     "--input_model_dir", str(model_dir),
                     "--output_dir", str(infer_dir),
                     "--calc_infer_scores", "true"] + infer_additional_args
        infer_result = subprocess.run(infer_run,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
        # Log and Time
        print(f"infer returncode = {infer_result.returncode}")
        save_log(infer_dir, infer_result)
        save_time(infer_dir, infer_start)
        print(f"Finished IMPROVE scripts with {lca} for training.")

    print(f"Finished LCA with {params['dataset']} {split_name}.")

print(f"Finished LCA. Results are in {params['output_dir']}")

 


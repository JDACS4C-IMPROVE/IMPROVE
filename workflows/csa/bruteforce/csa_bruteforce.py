""" Bruteforce implementation of cross-study analysis workflow """

import os
import time
import subprocess
import warnings
from pathlib import Path

from improvelib.initializer.config import Config
from workflows.utils_workflows import save_log, save_time, check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_list
import csa_bruteforce_params_def


start_full_wf = time.time()
filepath = Path(__file__).resolve().parent

cfg = Config() 
params = cfg.initialize_parameters(
    section="CSA",
    pathToModelDir=filepath,
    default_config="csa_bruteforce_params.ini",
    additional_definitions=csa_bruteforce_params_def.additional_definitions
)

# Make output_dir
output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)

#Model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Specify dirs
MAIN_ML_DATA_DIR = output_dir / 'ml_data'
MAIN_MODEL_DIR = output_dir / 'models'
MAIN_INFER_DIR = output_dir / 'infer' 
print("Created directory names.")
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)

# Note! Here input_dir is the location of benchmark data
splits_dir = Path(params['input_dir']) / params['splits_dir']


# Prepares additional parameters
preprocess_additional_args = get_additional_parameters(params['preprocess_args'])
if 'input_supp_data_dir' in preprocess_additional_args:
    preprocess_additional_args['input_supp_data_dir'] = check_dir_path_or_model_scripts_dir(preprocess_additional_args['input_supp_data_dir'], params['model_scripts_dir'])
preprocess_additional_args = additional_parameters_dict_to_list(preprocess_additional_args)

train_additional_args = get_additional_parameters(params['train_args'])
train_additional_args = additional_parameters_dict_to_list(train_additional_args)

infer_additional_args = get_additional_parameters(params['infer_args'])
infer_additional_args = additional_parameters_dict_to_list(infer_additional_args)

# ===============================================================
###  Generate CSA results 
# ===============================================================

print("source_datasets:", params["source_datasets"])
print("target_datasets:", params["target_datasets"])
print("split_nums:", params["split_nums"])

for source_data_name in params["source_datasets"]:
    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(params["split_nums"]) == 0:
        # Get all splits
        split_files = list((splits_dir).glob(f"{source_data_name}_split_*.txt"))
        params["split_nums"] = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        params["split_nums"] = sorted(set(params["split_nums"]))
    else:
        # Use the specified splits
        split_files = []
        for s in params["split_nums"]:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]

    for split in params["split_nums"]:
        print(f"Split id {split} out of {len(params['split_nums'])} splits.")
        # this needs to be a function
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        for phase in ["train", "val", "test"]:
            fname = f"{source_data_name}_split_{split}_{phase}.txt"
            if fname not in "\t".join(files_joined):
                warnings.warn(f"The {phase} split file {fname} is missing (continue to next split)")
                continue

        for target_data_name in params["target_datasets"]:
            if params["only_cross_study"] and (source_data_name == target_data_name):
                continue # only cross-study
            print(f"Source data: {source_data_name}")
            print(f"Target data: {target_data_name}")

            # Create dirs
            ml_data_dir = MAIN_ML_DATA_DIR / f"{source_data_name}-{target_data_name}" / f"split_{split}"
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}" / f"split_{split}"
            infer_dir = MAIN_INFER_DIR / f"{source_data_name}-{target_data_name}" / f"split_{split}"

            # Create split file names
            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ### PREPROCESS
            print(f"Preprocessing with train_split_file {train_split_file}, val_split_file {val_split_file}, and test_split_file {test_split_file}.")
            preprocess_run = ["python", preprocess_python_script,
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  "--input_dir", params['input_dir'], 
                  "--output_dir", str(ml_data_dir)] + preprocess_additional_args
            preprocess_start = time.time()
            preprocess_result = subprocess.run(preprocess_run,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            # Log and Time
            print(f"preprocess returncode = {preprocess_result.returncode}")
            save_log(ml_data_dir, preprocess_result)
            save_time(ml_data_dir, preprocess_start)

            ### TRAIN
            # Train a single model for a given [source, split] pair
            if model_dir.exists() is False:
                print(f"Train with ml_data_dir {ml_data_dir} and model_dir {model_dir}")
                train_run = ["python", train_python_script,
                    "--input_dir", str(ml_data_dir),
                    "--output_dir", str(model_dir)] + train_additional_args
                train_start = time.time()
                train_result = subprocess.run(train_run,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                
                print(f"train returncode = {train_result.returncode}")
                save_log(model_dir, train_result)
                save_time(model_dir, train_start)

            ### INFER
            start_infer = time.time()
            print(f"Infer with ml_data_dir {ml_data_dir}, model_dir {model_dir}, and infer_dir {infer_dir}.")
            infer_run = ["python", infer_python_script,
                "--input_data_dir", str(ml_data_dir),
                "--input_model_dir", str(model_dir),
                "--output_dir", str(infer_dir),
                "--calc_infer_scores", "true"] + infer_additional_args
            infer_start = time.time()
            infer_result = subprocess.run(infer_run,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            
            # Log and Time
            print(f"infer returncode = {infer_result.returncode}")
            save_log(infer_dir, infer_result)
            save_time(infer_dir, infer_start)



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
save_time(output_dir, start_full_wf)
print('Finished full cross-study run.')



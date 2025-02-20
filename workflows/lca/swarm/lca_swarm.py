import os
import glob
import json
import time
from pathlib import Path
import lca_swarm_params_def
from improvelib.initializer.config import Config

# parameters
filepath = Path(__file__).resolve().parent
cfg = Config() 
params = cfg.initialize_parameters(
    section="LCA",
    pathToModelDir=filepath,
    default_config="lca_bruteforce_params.ini",
    additional_definitions=lca_swarm_params_def.additional_definitions
)

output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)


#Model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Prefix string for swarm files
prefix = f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {params['model_scripts_dir']}/{params['model_environment']} ; export PYTHONPATH=../../../../IMPROVE ; "

# Specify dirs
MAIN_ML_DATA_DIR = output_dir / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = output_dir / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = output_dir / 'infer' # output_dir infer
print("Created directory names.")
print("output_dir:  ", output_dir)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)


supp_data_dir = None
if params['input_supp_data_dir'] is not None:
    supp_data_dir = params['input_supp_data_dir']
    # check if input_supp_data_dir is a directory
    if not os.path.isdir(supp_data_dir):
        # if input_supp_data_dir isn't a directory, check if it's in model_scripts_dir
        supp_data_dir = os.path.join(params['model_scripts_dir'], supp_data_dir)
        if not os.path.isdir(supp_data_dir):
            print("Parameter input_supp_data_dir provided but not found at provided path or in model_scripts_dir.")
            supp_data_dir = None

preprocess_list = []
train_list = []
infer_list = []

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
        ### PREPROCESS
        ml_data_dir = MAIN_ML_DATA_DIR / split_name / lca_name
        if supp_data_dir is not None:
            preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(lca_train_path)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(ml_data_dir)} --y_col_name {str(params['y_col_name'])} --input_supp_data_dir {str(supp_data_dir)}"]
        else:
            preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(lca_train_path)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(ml_data_dir)} --y_col_name {str(params['y_col_name'])}"]
        preprocess_list = preprocess_list + preprocess_run

        ### TRAIN
        model_dir = MAIN_MODEL_DIR / split_name / lca_name
        if params["cuda_name"] is not None:
            if params['epochs'] is not None:
                train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --epochs {str(params['epochs'])} --cuda_name {params['cuda_name']} --y_col_name {str(params['y_col_name'])}"]
            else:
                train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --cuda_name {params['cuda_name']} --y_col_name {str(params['y_col_name'])}"]
        else:
            if params['epochs'] is not None:
                train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --epochs {str(params['epochs'])} --y_col_name {str(params['y_col_name'])}"]
            else:
                train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --y_col_name {str(params['y_col_name'])}"]
        train_list = train_list + train_run

        ### INFER
        infer_dir = MAIN_INFER_DIR / split_name / lca_name
        if params["cuda_name"] is not None:
            infer_run = [f"python {infer_python_script} --input_data_dir {str(ml_data_dir)} --input_model_dir {str(model_dir)} --output_dir {str(infer_dir)} --cuda_name {params['cuda_name']} --y_col_name {str(params['y_col_name'])} --calc_infer_scores true"]
        else:
            infer_run = [f"python {infer_python_script} --input_data_dir {str(ml_data_dir)} --input_model_dir {str(model_dir)} --output_dir {str(infer_dir)} --y_col_name {str(params['y_col_name'])} --calc_infer_scores true"]
        infer_list = infer_list + infer_run

if params['swarm_file_prefix'] is not None:
    swarm_file_prefix = params['swarm_file_prefix']
else:
    swarm_file_prefix = params['model_name'] + "_" + params['dataset'] + "_"

with open(params['output_swarmfile_dir'] + swarm_file_prefix + "preprocess.swarm", "w") as file:
    for item in preprocess_list:
        file.write(prefix + item + "\n")

with open(params['output_swarmfile_dir'] + swarm_file_prefix + "train.swarm", "w") as file:
    for item in train_list:
        file.write(prefix + item + "\n")

with open(params['output_swarmfile_dir'] + swarm_file_prefix + "infer.swarm", "w") as file:
    for item in infer_list:
        file.write(prefix + item + "\n")

print(f"Finished swarm files. Swarm files are in {params['output_swarmfile_dir']} and prefixed with {swarm_file_prefix}. Results will be in {params['output_dir']}")

 


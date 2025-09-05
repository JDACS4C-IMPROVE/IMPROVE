import os
from pathlib import Path
import lca_swarm_params_def
from improvelib.initializer.config import Config
from workflows.utils_workflows import check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_string

# Set up parameters
filepath = Path(__file__).resolve().parent
cfg = Config() 
params = cfg.initialize_parameters(
    section="LCA",
    pathToModelDir=filepath,
    default_config="lca_swarm_params.ini",
    additional_definitions=lca_swarm_params_def.additional_definitions
)

# Make output_dir
output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)


# Create names for model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Prefix string for swarm files
if params['swarm_prefix'] is not None:
    prefix = params['swarm_prefix']
else:
    model_env = check_dir_path_or_model_scripts_dir(params['model_environment'], params['model_scripts_dir'])
    prefix = f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {model_env} ; export PYTHONPATH=../../../../IMPROVE ; "

# Specify dirs
MAIN_ML_DATA_DIR = output_dir / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = output_dir / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = output_dir / 'infer' # output_dir infer
print("Created directory names.")
print("output_dir:  ", output_dir)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)

# Prepare additional parameters
preprocess_additional_args = get_additional_parameters(params['preprocess_args'])
if 'input_supp_data_dir' in preprocess_additional_args:
    preprocess_additional_args['input_supp_data_dir'] = check_dir_path_or_model_scripts_dir(preprocess_additional_args['input_supp_data_dir'], params['model_scripts_dir'])
preprocess_additional_args = additional_parameters_dict_to_string(preprocess_additional_args)

train_additional_args = get_additional_parameters(params['train_args'])
train_additional_args = additional_parameters_dict_to_string(train_additional_args)

infer_additional_args = get_additional_parameters(params['infer_args'])
infer_additional_args = additional_parameters_dict_to_string(infer_additional_args)


# Create lists for swarm commands
preprocess_list = []
train_list = []
infer_list = []

## Loop through all splits specified in the parameters
for split_num in params['split_nums']:

    split_name = params['split_type'] + "_" + split_num
    print(f"Running LCA with {params['dataset']} {split_name}...")

    # Determine files for training shards from the lca_splits_dir
    lca_split_paths = list(Path(params['lca_splits_dir']).glob(f"{params['dataset']}_{params['split_type']}_{split_num}_sz_*.txt"))
    lca_split_files = [os.path.basename(x) for x in lca_split_paths]
    print(f"Running LCA on {len(lca_split_files)} shards with the following training splits:", lca_split_files)

    # Set val and test names
    val_split_file = f"{params['dataset']}_{params['split_type']}_{split_num}_val.txt"
    test_split_file = f"{params['dataset']}_{params['split_type']}_{split_num}_test.txt"

    
    ## Loop through all shards for the specified split
    for lca in lca_split_files:
        lca_train_path = params['lca_splits_dir'] + '/' + lca
        lca_name = "sz_" + lca.split('.')[0].split('_')[4]
        ### PREPROCESS
        ml_data_dir = MAIN_ML_DATA_DIR / split_name / lca_name
        preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(lca_train_path)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(ml_data_dir)} " + preprocess_additional_args]
        preprocess_list = preprocess_list + preprocess_run

        ### TRAIN
        model_dir = MAIN_MODEL_DIR / split_name / lca_name
        train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)}" + train_additional_args]
        train_list = train_list + train_run

        ### INFER
        infer_dir = MAIN_INFER_DIR / split_name / lca_name
        infer_run = [f"python {infer_python_script} --input_data_dir {str(ml_data_dir)} --input_model_dir {str(model_dir)} --output_dir {str(infer_dir)} --calc_infer_scores true" + infer_additional_args]
        infer_list = infer_list + infer_run

# Determine prefix for swarm file name
if params['swarm_file_prefix'] is not None:
    swarm_file_prefix = params['swarm_file_prefix']
else:
    swarm_file_prefix = params['model_name'] + "_" + params['dataset'] + "_" + params['split_type'] + "_"

# Save lists of swarm commands to file
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

 


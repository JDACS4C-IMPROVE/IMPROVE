import os
from pathlib import Path

import csa_swarm_params_def
from improvelib.initializer.config import Config
from workflows.utils_workflows import check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_string


# parameters
filepath = Path(__file__).resolve().parent
cfg = Config() 
params = cfg.initialize_parameters(
    section="CSA",
    pathToModelDir=filepath,
    default_config="csa_swarm_params.ini",
    additional_definitions=csa_swarm_params_def.additional_definitions
)

# Make output dir
output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)

# Prefix string for swarm files
if params['swarm_prefix'] is not None:
    prefix = params['swarm_prefix']
else:
    model_env = check_dir_path_or_model_scripts_dir(params['model_environment'], params['model_scripts_dir'])
    prefix = f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {model_env} ; export PYTHONPATH=../../../../IMPROVE ; "

#Model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Specify dirs
MAIN_CSA_OUTDIR = Path(params["output_dir"]) 
MAIN_ML_DATA_DIR = MAIN_CSA_OUTDIR / 'ml_data' 
MAIN_MODEL_DIR = MAIN_CSA_OUTDIR / 'models' 
MAIN_INFER_DIR = MAIN_CSA_OUTDIR / 'infer'

print("Created directory names.")
print("MAIN_CSA_OUTDIR:  ", MAIN_CSA_OUTDIR)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)

# Note! Here input_dir is the location of benchmark data
splits_dir = Path(params['input_dir']) / "splits"


# Prepare additional parameters
preprocess_additional_args = get_additional_parameters(params['preprocess_args'])
if 'input_supp_data_dir' in preprocess_additional_args:
    preprocess_additional_args['input_supp_data_dir'] = check_dir_path_or_model_scripts_dir(preprocess_additional_args['input_supp_data_dir'], params['model_scripts_dir'])
preprocess_additional_args = additional_parameters_dict_to_string(preprocess_additional_args)

train_additional_args = get_additional_parameters(params['train_args'])
train_additional_args = additional_parameters_dict_to_string(train_additional_args)

infer_additional_args = get_additional_parameters(params['infer_args'])
infer_additional_args = additional_parameters_dict_to_string(infer_additional_args)

# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================


print("source_datasets:", params["source_datasets"])
print("target_datasets:", params["target_datasets"])
print("split_nums:", params["split_nums"])
print("split_type:", params['split_type'])

preprocess_list = []
train_list = []
infer_list = []

for source_data_name in params["source_datasets"]:
    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(params["split_nums"]) == 0:
        # Get all splits
        split_files = list((splits_dir).glob(f"{source_data_name}_{params['split_type']}_*.txt"))
        params["split_nums"] = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        params["split_nums"] = sorted(set(params["split_nums"]))
    else:
        # Use the specified splits
        split_files = []
        for s in params["split_nums"]:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_{params['split_type']}_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]
    print("FILES JOINED:", files_joined)

    for split in params["split_nums"]:
        not_trained_yet = True
        print(f"Split id {split} out of {len(params['split_nums'])} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        for phase in ["train", "val", "test"]:
            fname = f"{source_data_name}_{params['split_type']}_{split}_{phase}.txt"
            if fname not in "\t".join(files_joined):
                print(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                continue

        for target_data_name in params["target_datasets"]:
            if params["only_cross_study"] and (source_data_name == target_data_name):
                continue # only cross-study

            # set dirs
            ml_data_dir = MAIN_ML_DATA_DIR / f"{source_data_name}-{target_data_name}" / f"{params['split_type']}_{split}"
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}" / f"{params['split_type']}_{split}"
            infer_dir = MAIN_INFER_DIR / f"{source_data_name}-{target_data_name}" / f"{params['split_type']}_{split}"

            # set split file names
            train_split_file = f"{source_data_name}_{params['split_type']}_{split}_train.txt"
            val_split_file = f"{source_data_name}_{params['split_type']}_{split}_val.txt"
            if source_data_name == target_data_name: # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_{params['split_type']}_{split}_test.txt"
            else: # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # Preprocess
            preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(train_split_file)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(ml_data_dir)}" + preprocess_additional_args]
            preprocess_list = preprocess_list + preprocess_run
            # Train a single model for a given [source, split] pair
            if not_trained_yet:
                train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)}" + train_additional_args]
                train_list = train_list + train_run
                not_trained_yet = False
            # Infer
            infer_run = [f"python {infer_python_script} --input_data_dir {str(ml_data_dir)} --input_model_dir {str(model_dir)} --output_dir {str(infer_dir)} --calc_infer_scores true" + infer_additional_args]
            infer_list = infer_list + infer_run

# Determine prefix for swarm file name
if params['swarm_file_prefix'] is not None:
    swarm_file_prefix = params['swarm_file_prefix']
else:
    swarm_file_prefix = params['model_name'] + "_" + params['split_type'] + "_"

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



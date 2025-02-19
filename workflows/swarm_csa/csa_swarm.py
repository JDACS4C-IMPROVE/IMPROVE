import os
import warnings
from pathlib import Path

import csa_swarm_params_def
from improvelib.initializer.config import Config

# parameters
filepath = Path(__file__).resolve().parent
cfg = Config() 
params = cfg.initialize_parameters(
    section="CSA",
    pathToModelDir=filepath,
    default_config="csa_swarm_params.ini",
    additional_definitions=csa_swarm_params_def.additional_definitions
)

output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)

# Prefix string for swarm files
prefix = f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {params['model_scripts_dir']}/{params['model_environment']} ; export PYTHONPATH=../../../../IMPROVE ; "


#Model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Specify dirs
y_col_name = params['y_col_name']
MAIN_CSA_OUTDIR = Path(params["output_dir"]) # main output dir
MAIN_ML_DATA_DIR = MAIN_CSA_OUTDIR / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = MAIN_CSA_OUTDIR / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = MAIN_CSA_OUTDIR / 'infer' # output_dir infer

print("Created directory names.")
print("MAIN_CSA_OUTDIR:  ", MAIN_CSA_OUTDIR)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)

# Note! Here input_dir is the location of benchmark data
splits_dir = Path(params['input_dir']) / "splits"
print("Created splits path.")
print("splits_dir: ", splits_dir)

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


# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================


print("source_datasets:", params["source_datasets"])
print("target_datasets:", params["target_datasets"])
print("split_nums:", params["split_nums"])

preprocess_list = []
train_list = []
infer_list = []

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
    print("FILES JOINED:", files_joined)

    for split in params["split_nums"]:
        not_trained_yet = True
        print(f"Split id {split} out of {len(params['split_nums'])} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        for phase in ["train", "val", "test"]:
            fname = f"{source_data_name}_split_{split}_{phase}.txt"
            if fname not in "\t".join(files_joined):
                print(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                continue

        for target_data_name in params["target_datasets"]:
            if params["only_cross_study"] and (source_data_name == target_data_name):
                continue # only cross-study

            # set dirs
            ml_data_dir = MAIN_ML_DATA_DIR / f"{source_data_name}-{target_data_name}" / f"split_{split}"
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}" / f"split_{split}"
            infer_dir = MAIN_INFER_DIR / f"{source_data_name}-{target_data_name}" / f"split_{split}"

            # set split file names
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            if source_data_name == target_data_name: # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else: # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # Preprocess
            if supp_data_dir is not None:
                preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(train_split_file)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(ml_data_dir)} --y_col_name {str(y_col_name)} --input_supp_data_dir {str(supp_data_dir)}"]
            else:
                preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(train_split_file)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(ml_data_dir)} --y_col_name {str(y_col_name)}"]
            preprocess_list = preprocess_list + preprocess_run
            # Train a single model for a given [source, split] pair
            if not_trained_yet:
                if params["cuda_name"] is not None:
                    if params['epochs'] is not None:
                        train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --epochs {str(params['epochs'])} --cuda_name {params['cuda_name']} --y_col_name {y_col_name}"]
                    else:
                        train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --cuda_name {params['cuda_name']} --y_col_name {y_col_name}"]
                else:
                    if params['epochs'] is not None:
                        train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --epochs {str(params['epochs'])} --y_col_name {y_col_name}"]
                    else:
                        train_run = [f"python {train_python_script} --input_dir {str(ml_data_dir)} --output_dir {str(model_dir)} --y_col_name {y_col_name}"]
                train_list = train_list + train_run
                not_trained_yet = False
            # Infer
            if params["cuda_name"] is not None:
                infer_run = [f"python {infer_python_script} --input_data_dir {str(ml_data_dir)} --input_model_dir {str(model_dir)} --output_dir {str(infer_dir)} --cuda_name {params['cuda_name']} --y_col_name {y_col_name} --calc_infer_scores true"]
            else:
                infer_run = [f"python {infer_python_script} --input_data_dir {str(ml_data_dir)} --input_model_dir {str(model_dir)} --output_dir {str(infer_dir)} --y_col_name {y_col_name} --calc_infer_scores true"]
            infer_list = infer_list + infer_run


if params['swarm_file_prefix'] is not None:
    swarm_file_prefix = params['swarm_file_prefix']
else:
    swarm_file_prefix = params['model_name'] + "_" 

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



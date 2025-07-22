""" Bruteforce implementation of randomization workflow """

import os
import time
import subprocess
from pathlib import Path
import itertools
from ast import literal_eval
import shlex

from improvelib.initializer.config import Config
from workflows.utils_workflows import check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_string
import randomize_swarm_params_def


filepath = Path(__file__).resolve().parent

cfg = Config() 
params = cfg.initialize_parameters(
    section="RANDOMIZE",
    pathToModelDir=filepath,
    default_config="randomize_swarm_params.ini",
    additional_definitions=randomize_swarm_params_def.additional_definitions
)

# Make output_dir
main_output_dir = Path(params['output_dir'])
if main_output_dir.exists() is False:
    os.makedirs(main_output_dir, exist_ok=True)

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


# Prepares additional parameters
preprocess_additional_args = get_additional_parameters(params['preprocess_args'])
if 'input_supp_data_dir' in preprocess_additional_args:
    preprocess_additional_args['input_supp_data_dir'] = check_dir_path_or_model_scripts_dir(preprocess_additional_args['input_supp_data_dir'], params['model_scripts_dir'])
preprocess_additional_args = additional_parameters_dict_to_string(preprocess_additional_args)

train_additional_args = get_additional_parameters(params['train_args'])
train_additional_args = additional_parameters_dict_to_string(train_additional_args)

infer_additional_args = get_additional_parameters(params['infer_args'])
infer_additional_args = additional_parameters_dict_to_string(infer_additional_args)

# ===============================================================
###  Generate Randomization results 
# ===============================================================

preprocess_list = []
train_list = []
infer_list = []

print("DATASETS:", params["datasets"])
print("SPLIT_TYPES:", params["split_types"])
print("SPLITS:", params["split_nums"])

for dataset in params['datasets']:
    for split_type in params['split_types']:
        for split in params['split_nums']:
            # Create split file names
            test_split_file = f"{dataset}_{split_type}_{split}_test.txt"
            train_split_file = f"{dataset}_{split_type}_{split}_train.txt"
            val_split_file = f"{dataset}_{split_type}_{split}_val.txt"
            #files_dict = {'cell_transcriptomics_file': ['default', 'cell_shuffle_full_1.tsv', 'cell_shuffle_full_2.tsv'], 'drug_mordred_file': ['default', 'drug_mordred_shuffle_full_1.tsv']}
            randomized_data = literal_eval(params['randomized_data'])
            list_of_files = list(randomized_data.values())
            list_of_file_types = list(randomized_data.keys())
            file_combos = list(itertools.product(*list_of_files))
            list_of_input_data = []
            list_of_output_dirs = []
            for file_tuple in file_combos:
                input_data_string = ""
                folder_string = ""
                for n, file_type in enumerate(list_of_file_types):
                    file_value = file_tuple[n]
                    if file_value == 'default':
                        str_to_append = ""
                        file_rep = 'default'
                    else:
                        str_to_append = f" --{file_type} {params['randomized_data_dir']}/{file_value}"
                        file_rep = file_value.split('.')[0]
                    input_data_string = input_data_string + str_to_append
                    folder_string = folder_string + file_rep + '-'
                list_of_input_data = list_of_input_data + [input_data_string]
                folder_string = folder_string[:-1]
                list_of_output_dirs = list_of_output_dirs + [folder_string]

            print(list_of_input_data)
            print(list_of_output_dirs)

            for n in range(len(list_of_input_data)):
                folder_name = list_of_output_dirs[n]
                # Create dirs
                output_dir = main_output_dir/ dataset / split_type / f"split_{split}" / folder_name
                data_string = list_of_input_data[n]
                ### PREPROCESS
                print(f"Dataset {dataset}, split type {split_type}, split {split}")
                print(f"Data string {data_string}")
                print(f"Saves to {output_dir}")
                print(f"Preprocessing.")
                preprocess_run = [f"python {preprocess_python_script} --train_split_file {str(train_split_file)} --val_split_file {str(val_split_file)} --test_split_file {str(test_split_file)} --input_dir {params['input_dir']} --output_dir {str(output_dir)}" + data_string + preprocess_additional_args]
                preprocess_list = preprocess_list + preprocess_run
                ### TRAIN
                print(f"Training.")
                train_run = [f"python {train_python_script} --input_dir {str(output_dir)} --output_dir {str(output_dir)}" + train_additional_args]
                train_list = train_list + train_run
                ### INFER
                print(f"Inference.")
                infer_run = [f"python {infer_python_script} --input_data_dir {str(output_dir)} --input_model_dir {str(output_dir)} --output_dir {str(output_dir)} --calc_infer_scores true" + infer_additional_args]
                infer_list = infer_list + infer_run



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Determine prefix for swarm file name
if params['swarm_file_prefix'] is not None:
    swarm_file_prefix = params['swarm_file_prefix']
else:
    swarm_file_prefix = params['model_name'] + "_" 

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




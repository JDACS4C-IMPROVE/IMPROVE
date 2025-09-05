""" Bruteforce implementation of randomization workflow """

import os
import time
import subprocess
from pathlib import Path
import itertools
from ast import literal_eval
import shlex

from improvelib.initializer.config import Config
from workflows.utils_workflows import save_log, save_time, check_dir_path_or_model_scripts_dir, get_additional_parameters, additional_parameters_dict_to_list
import randomize_bruteforce_params_def



start_full_wf = time.time()
filepath = Path(__file__).resolve().parent

cfg = Config() 
params = cfg.initialize_parameters(
    section="RANDOMIZE",
    pathToModelDir=filepath,
    default_config="randomize_bruteforce_params.ini",
    additional_definitions=randomize_bruteforce_params_def.additional_definitions
)

# Make output_dir
main_output_dir = Path(params['output_dir'])
if main_output_dir.exists() is False:
    os.makedirs(main_output_dir, exist_ok=True)

#Model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")


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
                preprocess_run = ["python", preprocess_python_script,
                    "--train_split_file", str(train_split_file),
                    "--val_split_file", str(val_split_file),
                    "--test_split_file", str(test_split_file),
                    "--input_dir", params['input_dir'], 
                    "--output_dir", str(output_dir)] + shlex.split(data_string) + preprocess_additional_args
                preprocess_start = time.time()
                preprocess_result = subprocess.run(preprocess_run,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                # Log and Time
                print(f"preprocess returncode = {preprocess_result.returncode}")
                save_log(output_dir, preprocess_result, prefix='preprocess_')
                save_time(output_dir, preprocess_start, prefix='preprocess_')

                ### TRAIN
                print(f"Training.")
                train_run = ["python", train_python_script,
                    "--input_dir", str(output_dir),
                    "--output_dir", str(output_dir)] + train_additional_args
                train_start = time.time()
                train_result = subprocess.run(train_run,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                
                print(f"train returncode = {train_result.returncode}")
                save_log(output_dir, train_result, prefix='train_')
                save_time(output_dir, train_start, prefix='train_')
                ### INFER
                start_infer = time.time()
                print(f"Inference.")
                infer_run = ["python", infer_python_script,
                    "--input_data_dir", str(output_dir),
                    "--input_model_dir", str(output_dir),
                    "--output_dir", str(output_dir),
                    "--calc_infer_scores", "true"] + infer_additional_args
                infer_start = time.time()
                infer_result = subprocess.run(infer_run,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                
                # Log and Time
                print(f"infer returncode = {infer_result.returncode}")
                save_log(output_dir, infer_result, prefix='infer_')
                save_time(output_dir, infer_start, prefix='infer_')



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
save_time(output_dir, start_full_wf, prefix='workflow_')
print('Finished full randomize run.')



# Standard library imports
import os
import json
import time
from pathlib import Path
from ast import literal_eval


def save_log(dir, result, prefix=""):
    result_file_name_stdout = dir / f'{prefix}logs.txt'
    if dir.exists() is False: 
        os.makedirs(dir, exist_ok=True)
    with open(result_file_name_stdout, 'w') as file:
        file.write(result.stdout)

def save_time(dir, start, prefix=""):
    time_diff = time.time() - start
    hours = int(time_diff // 3600)
    minutes = int((time_diff % 3600) // 60)
    seconds = time_diff % 60
    time_diff_dict = {'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds}
    with open(Path(dir) / f'{prefix}runtime.json', 'w') as json_file:
        json.dump(time_diff_dict, json_file, indent=4)


def check_dir_path_or_model_scripts_dir(dir, dir_to_check):
    dir_path = None
    if dir is None:
        dir_path = None
    elif os.path.isdir(dir):
        dir_path = dir
    elif os.path.isdir(os.path.join(dir_to_check, dir)):
        dir_path = os.path.join(dir_to_check, dir)
    else:
        raise ValueError(f"Parameter input_supp_data_dir provided but not found at provided path or in {dir_to_check}.")
    return dir_path

def get_additional_parameters(param_list_str):
    param_dict = literal_eval(param_list_str)
    #param_dict = dict(param_list)
    return param_dict

def additional_parameters_dict_to_list(param_dict):
    prefix_dict = {f"--{key}": value for key, value in param_dict.items()}
    arg_list = []
    for key, value in prefix_dict.items():
        arg_list = arg_list + [str(key)]
        arg_list = arg_list + [str(value)]
    return arg_list

def additional_parameters_dict_to_string(param_dict):
    prefix_dict = {f"--{key}": value for key, value in param_dict.items()}
    arg_string = " "
    for key, value in prefix_dict.items():
        arg_string = arg_string + f"{key}" + " "
        if isinstance(value, list):
            arg_string = arg_string + f"\"{value}\"" + " "
        else:
            arg_string = arg_string + f"{value}" + " "
    return arg_string
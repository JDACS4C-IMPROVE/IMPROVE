""" Bruteforce implementation of cross-study analysis workflow """

import json
import os
import time
import subprocess
import warnings
from pathlib import Path

import pandas as pd


# IMPROVE imports
# from improvelib.initializer.config import Config
# from improvelib.initializer.stage_config import PreprocessConfig, TrainConfig, InferConfig
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
# from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
# from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm
from csa_bruteforce_params_def import csa_bruteforce_params
from improvelib.utils import Timer

start_full_wf = time.time()

'''
def build_split_fname(source: str, split: int, phase: str):
    """ Build split file name. If file does not exist continue """
    return f"{source_data_name}_split_{split}_{phase}.txt"


def save_captured_output(result,
                         process,
                         MAIN_LOG_DIR,
                         source_data_name,
                         target_data_name,
                         split):
    result_file_name_stdout = MAIN_LOG_DIR / \
        f"{source_data_name}-{target_data_name}-{split}-{process}-log.txt"
    with open(result_file_name_stdout, 'w') as file:
        file.write(result.stdout)
    return True
'''


filepath = Path(__file__).resolve().parent

print(f"File path: {filepath}")


# ===============================================================
###  CSA settings
# ===============================================================

cfg = DRPPreprocessConfig()
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_bruteforce_params.ini",
    additional_definitions=csa_bruteforce_params,
    required=None
)
print("Loaded params")

# Model scripts
#model_name = params["model_name"]
#preprocess_python_script = f'{model_name}_preprocess_improve.py'
#train_python_script = f'{model_name}_train_improve.py'
#infer_python_script = f'{model_name}_infer_improve.py'

#Model scripts
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")

# Specify dirs
y_col_name = params['y_col_name']
MAIN_CSA_OUTDIR = Path(params["csa_outdir"]) # main output dir
MAIN_ML_DATA_DIR = MAIN_CSA_OUTDIR / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = MAIN_CSA_OUTDIR / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = MAIN_CSA_OUTDIR / 'infer' # output_dir infer
MAIN_LOG_DIR = MAIN_CSA_OUTDIR / 'logs'
frm.create_outdir(MAIN_LOG_DIR)
print("Created directory names.")
print("MAIN_CSA_OUTDIR:  ", MAIN_CSA_OUTDIR)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)
print("MAIN_LOG_DIR:     ", MAIN_LOG_DIR)
# Note! Here input_dir is the location of benchmark data
splits_dir = Path(params['input_dir']) / params['splits_dir']
print("Created splits path.")
print("splits_dir: ", splits_dir)

try:
    # check if input_supp_data_dir provided
    supp_data_dir = params['input_supp_data_dir']
    # check if input_supp_data_dir is a directory
    if not os.path.isdir(supp_data_dir):
        # if input_supp_data_dir isn't a directory, check if it's in model_scripts_dir
        supp_data_dir = os.path.join(params['model_scripts_dir'],supp_data_dir)
        if not os.path.isdir(supp_data_dir):
            print("Parameter input_supp_data_dir provided but not found at provided bath or in model_scripts_dir.")
except KeyError:
    # if no input_supp_data_dir provided, set to empty string
    supp_data_dir = ""

# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================

# timer = Timer()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
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
        # num_splits = 1
    else:
        # Use the specified splits
        split_files = []
        for s in params["split_nums"]:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]

    # --------------------
    # Preprocess and Train
    # --------------------
    for split in params["split_nums"]:
        print(f"Split id {split} out of {len(params['split_nums'])} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        for phase in ["train", "val", "test"]:
            fname = f"{source_data_name}_split_{split}_{phase}.txt"
            if fname not in "\t".join(files_joined):
                warnings.warn(f"\nThe {phase} split file {fname} is missing \
                              (continue to next split)")
                continue

        for target_data_name in params["target_datasets"]:
            if params["only_cross_study"] and (source_data_name == target_data_name):
                continue # only cross-study
            print(f"\nSource data: {source_data_name}")
            print(f"Target data: {target_data_name}")

            ml_data_dir = MAIN_ML_DATA_DIR / \
                f"{source_data_name}-{target_data_name}" / f"split_{split}"
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}" / f"split_{split}"
            infer_dir = MAIN_INFER_DIR / \
                f"{source_data_name}-{target_data_name}" / f"split_{split}"

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            start_preprocess = time.time()
            # timer_preprocess = Timer()
            print("\nPreprocessing")
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            print(f"train_split_file: {train_split_file}")
            print(f"val_split_file:   {val_split_file}")
            print(f"test_split_file:  {test_split_file}")
            preprocess_run = ["python", preprocess_python_script,
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  "--input_dir", params['input_dir'], # str("./csa_data/raw_data"),
                  "--output_dir", str(ml_data_dir),
                  "--y_col_name", str(y_col_name),
                  "--input_supp_data_dir", str(supp_data_dir)
            ]
            result = subprocess.run(preprocess_run,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            
            # Logger
            print(f"returncode = {result.returncode}")
            result_file_name_stdout = ml_data_dir / 'logs.txt'
            if ml_data_dir.exists() is False: 
                os.makedirs(ml_data_dir, exist_ok=True)
            with open(result_file_name_stdout, 'w') as file:
                file.write(result.stdout)

            # Timer
            time_diff = time.time() - start_preprocess
            hours = int(time_diff // 3600)
            minutes = int((time_diff % 3600) // 60)
            seconds = time_diff % 60
            time_diff_dict = {'hours': hours,
                            'minutes': minutes,
                            'seconds': seconds}
            dir_to_save = ml_data_dir
            filename = 'runtime.json'
            with open(Path(dir_to_save) / filename, 'w') as json_file:
                json.dump(time_diff_dict, json_file, indent=4)


            # print(f"returncode = {result.returncode}")
            # save_captured_output(result, "preprocess", MAIN_LOG_DIR,
            #                      source_data_name, target_data_name, split)
            # tt = timer_preprocess.display_timer(print_fn)
            # extra_dict = {"source_data": source_data_name,
            #               "target_data": target_data_name,
            #               "split": split}
            # timer_preprocess.save_timer(ml_data_dir, extra_dict=extra_dict)

            # p2 (p1): Train model
            # Train a single model for a given [source, split] pair
            # Train using train samples and early stop using val samples
            if model_dir.exists() is False:
                start_train = time.time()
                # timer_train = Timer()
                print("\nTrain")
                print(f"ml_data_dir: {ml_data_dir}")
                print(f"model_dir:   {model_dir}")
                if params["uses_cuda_name"]:
                    train_run = ["python", train_python_script,
                        "--input_dir", str(ml_data_dir),
                        "--output_dir", str(model_dir),
                        "--epochs", str(params["epochs"]),  # DL-specific
                        "--cuda_name", params["cuda_name"], # DL-specific
                        "--y_col_name", y_col_name
                    ]
                else:
                    train_run = ["python", train_python_script,
                        "--input_dir", str(ml_data_dir),
                        "--output_dir", str(model_dir),
                        "--epochs", str(params["epochs"]),  # DL-specific
                        "--y_col_name", y_col_name
                    ]
                result = subprocess.run(train_run,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
                
                # Logger
                print(f"returncode = {result.returncode}")
                result_file_name_stdout = model_dir / 'logs.txt'
                if model_dir.exists() is False: # If subprocess fails, model_dir may not be created and we need to write the log files in model_dir
                    os.makedirs(model_dir, exist_ok=True)
                with open(result_file_name_stdout, 'w') as file:
                    file.write(result.stdout)

                # Timer
                time_diff = time.time() - start_train
                hours = int(time_diff // 3600)
                minutes = int((time_diff % 3600) // 60)
                seconds = time_diff % 60
                time_diff_dict = {'hours': hours,
                                'minutes': minutes,
                                'seconds': seconds}
                dir_to_save = model_dir
                filename = 'runtime.json'
                with open(Path(dir_to_save) / filename, 'w') as json_file:
                    json.dump(time_diff_dict, json_file, indent=4)






                # print(f"returncode = {result.returncode}")
                # save_captured_output(result, "train", MAIN_LOG_DIR,
                #                      source_data_name, "none", split)
                # tt = timer_train.display_timer(print_fn)
                # extra_dict = {"source_data": source_data_name, "split": split}
                # timer_train.save_timer(model_dir, extra_dict=extra_dict)

            # Infer
            # p3 (p1, p2): Inference
            start_infer = time.time()
            # timer_infer = Timer()
            print("\nInfer")
            print(f"ml_data_dir: {ml_data_dir}")
            print(f"model_dir:   {model_dir}")
            print(f"infer_dir:   {infer_dir}")
            if params["uses_cuda_name"]:
                infer_run = ["python", infer_python_script,
                    "--input_data_dir", str(ml_data_dir),
                    "--input_model_dir", str(model_dir),
                    "--output_dir", str(infer_dir),
                    "--cuda_name", params["cuda_name"], # DL-specific
                    "--y_col_name", y_col_name,
                    "--calc_infer_scores", "true"
                ]
            else:
                infer_run = ["python", infer_python_script,
                    "--input_data_dir", str(ml_data_dir),
                    "--input_model_dir", str(model_dir),
                    "--output_dir", str(infer_dir),
                    "--y_col_name", y_col_name,
                    "--calc_infer_scores", "true"
                ]
            result = subprocess.run(infer_run,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            
            # Logger
            print(f"returncode = {result.returncode}")
            result_file_name_stdout = infer_dir / 'logs.txt'
            if infer_dir.exists() is False: 
                os.makedirs(infer_dir, exist_ok=True)
            with open(result_file_name_stdout, 'w') as file:
                file.write(result.stdout)

            # Timer
            time_diff = time.time() - start_infer
            hours = int(time_diff // 3600)
            minutes = int((time_diff % 3600) // 60)
            seconds = time_diff % 60
            time_diff_dict = {'hours': hours,
                            'minutes': minutes,
                            'seconds': seconds}
            dir_to_save = infer_dir
            filename = 'runtime.json'
            with open(Path(dir_to_save) / filename, 'w') as json_file:
                json.dump(time_diff_dict, json_file, indent=4)








            # print(f"returncode = {result.returncode}")
            # save_captured_output(result, "infer", MAIN_LOG_DIR,
            #                      source_data_name, target_data_name, split)
            # tt = timer_infer.display_timer(print_fn)
            # extra_dict = {"source_data": source_data_name,
            #               "target_data": target_data_name,
            #               "split": split}
            # timer_infer.save_timer(infer_dir, extra_dict=extra_dict)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Timer - full run
time_diff = time.time() - start_full_wf
hours = int(time_diff // 3600)
minutes = int((time_diff % 3600) // 60)
seconds = time_diff % 60
time_diff_dict = {'hours': hours,
                  'minutes': minutes,
                  'seconds': seconds}
dir_to_save = params['output_dir']
filename = 'full_runtime.json'
with open(Path(dir_to_save) / filename, 'w') as json_file:
    json.dump(time_diff_dict, json_file, indent=4)


print('Finished full cross-study run.')



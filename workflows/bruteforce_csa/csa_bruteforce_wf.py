""" Python implementation of cross-study analysis workflow """


import os
import subprocess
import warnings
from time import time
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


def build_split_fname(source: str, split: int, phase: str):
    """ Build split file name. If file does not exist continue """
    return f"{source_data_name}_split_{split}_{phase}.txt"

def save_captured_output(result, process, MAIN_LOG_DIR, source_data_name, target_data_name, split):
    result_file_name_stdout = MAIN_LOG_DIR / f"{source_data_name}-{target_data_name}-{split}-{process}-log.txt"
    with open(result_file_name_stdout, 'w') as file:
        file.write(result.stdout)




class Timer:
    """ Measure time. """
    def __init__(self):
        self.start = time()

    def timer_end(self):
        self.end = time()
        return self.end - self.start

    def display_timer(self, print_fn=print):
        time_diff = self.timer_end()
        if (time_diff) // 3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
        else:
            print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )


filepath = Path(__file__).resolve().parent

print_fn = print
print_fn(f"File path: {filepath}")

# ===============================================================
###  CSA settings
# ===============================================================


cfg = DRPPreprocessConfig() # TODO submit github issue; too many logs printed; is it necessary?
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_bruteforce_params.ini",
    additional_definitions=csa_bruteforce_params,
    required=None
)
print("Loaded params")

# Model scripts
model_name = params["model_name"]
preprocess_python_script = f'{model_name}_preprocess_improve.py'
train_python_script = f'{model_name}_train_improve.py'
infer_python_script = f'{model_name}_infer_improve.py'
print("Created script names")
# Specify dirs

y_col_name = params['y_col_name']
# maindir = Path(f"./{y_col_name}")
# maindir = Path(f"./0_{y_col_name}_improvelib") # main output dir
MAIN_CSA_OUTDIR = Path(params["csa_outdir"]) # main output dir
# Note! ML data and trained model should be saved to the same dir for inference script
MAIN_ML_DATA_DIR = MAIN_CSA_OUTDIR / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = MAIN_CSA_OUTDIR / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = MAIN_CSA_OUTDIR / 'infer' # output_dir infer
MAIN_LOG_DIR = MAIN_CSA_OUTDIR / 'logs'
frm.create_outdir(MAIN_LOG_DIR)
print("Created directory names")
print("MAIN_CSA_OUTDIR: ", MAIN_CSA_OUTDIR)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR: ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR: ", MAIN_INFER_DIR)
print("MAIN_LOG_DIR: ", MAIN_LOG_DIR)
# Note! Here input_dir is the location of benchmark data
# TODO Should we set input_dir (and output_dir) for each models scrit?
splits_dir = Path(params['input_dir']) / params['splits_dir']
print("Created splits path")
print("splits_dir: ", splits_dir)

source_datasets = params["source_datasets"]
target_datasets = params["target_datasets"]
only_cross_study = params["only_cross_study"]
split_nums = params["split_nums"]
epochs = params["epochs"]
cuda_name = params["cuda_name"]
print("internal params")


# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================

timer = Timer()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
print_fn(f"\nsource_datasets: {source_datasets}")
print_fn(f"target_datasets: {target_datasets}")
print_fn(f"split_nums:      {split_nums}")
# breakpoint()
for source_data_name in source_datasets:

    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((splits_dir).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
        # num_splits = 1
    else:
        # Use the specified splits
        split_files = []
        for s in split_nums:
            split_files.extend(list((splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    files_joined = [str(s) for s in split_files]

    # --------------------
    # Preprocess and Train
    # --------------------
    for split in split_nums:
        print_fn(f"Split id {split} out of {len(split_nums)} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        for phase in ["train", "val", "test"]:
            fname = build_split_fname(source_data_name, split, phase)
            if fname not in "\t".join(files_joined):
                warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
                continue

        for target_data_name in target_datasets:
            if only_cross_study and (source_data_name == target_data_name):
                continue # only cross-study
            print_fn(f"\nSource data: {source_data_name}")
            print_fn(f"Target data: {target_data_name}")

            ml_data_dir = MAIN_ML_DATA_DIR / f"{source_data_name}-{target_data_name}" / \
                f"split_{split}"
            model_dir = MAIN_MODEL_DIR / f"{source_data_name}" / f"split_{split}"
            infer_dir = MAIN_INFER_DIR / f"{source_data_name}-{target_data_name}" / \
                f"split_{split}" # AP

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # placeholder for LC
            timer_preprocess = Timer()
            print_fn("\nPreprocessing")
            train_split_file = f"{source_data_name}_split_{split}_train.txt"
            val_split_file = f"{source_data_name}_split_{split}_val.txt"
            print_fn(f"train_split_file: {train_split_file}")
            print_fn(f"val_split_file:   {val_split_file}")
            print_fn(f"test_split_file:  {test_split_file}")
            preprocess_run = ["python", preprocess_python_script,
                  "--train_split_file", str(train_split_file),
                  "--val_split_file", str(val_split_file),
                  "--test_split_file", str(test_split_file),
                  "--input_dir", params['input_dir'], # str("./csa_data/raw_data"),
                  "--output_dir", str(ml_data_dir),
                  "--y_col_name", str(y_col_name)
            ]
            result = subprocess.run(preprocess_run, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, universal_newlines=True)
            print(f"returncode = {result.returncode}")
            save_captured_output(result, "preprocess", MAIN_LOG_DIR, source_data_name, target_data_name, split)
            timer_preprocess.display_timer(print_fn)

            # p2 (p1): Train model
            # Train a single model for a given [source, split] pair
            # Train using train samples and early stop using val samples
            if model_dir.exists() is False:
                timer_train = Timer()
                print_fn("\nTrain")
                print_fn(f"ml_data_dir: {ml_data_dir}")
                print_fn(f"model_dir:   {model_dir}")
                if params["uses_cuda_name"]:
                    train_run = ["python", train_python_script,
                        "--input_dir", str(ml_data_dir),
                        "--output_dir", str(model_dir),
                        "--epochs", str(epochs),  # DL-specific
                        "--cuda_name", cuda_name, # DL-specific
                        "--y_col_name", y_col_name
                    ]
                else:
                    train_run = ["python", train_python_script,
                        "--input_dir", str(ml_data_dir),
                        "--output_dir", str(model_dir),
                        "--epochs", str(epochs),  # DL-specific
                        "--y_col_name", y_col_name
                    ]
                result = subprocess.run(train_run, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, universal_newlines=True)
                print(f"returncode = {result.returncode}")
                save_captured_output(result, "train", MAIN_LOG_DIR, source_data_name, "none", split)
                timer_train.display_timer(print_fn)

            # Infer
            # p3 (p1, p2): Inference
            timer_infer = Timer()
            print_fn("\nInfer")
            if params["uses_cuda_name"]:
                infer_run = ["python", infer_python_script,
                    "--input_data_dir", str(ml_data_dir),
                    "--input_model_dir", str(model_dir),
                    "--output_dir", str(infer_dir),
                    "--cuda_name", cuda_name, # DL-specific
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
            result = subprocess.run(infer_run, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, universal_newlines=True)
            print(f"returncode = {result.returncode}")
            save_captured_output(result, "infer", MAIN_LOG_DIR, source_data_name, target_data_name, split)
            timer_infer.display_timer(print_fn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

timer.display_timer(print_fn)
print_fn('Finished full cross-study run.')
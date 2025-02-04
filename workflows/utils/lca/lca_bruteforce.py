import os
import glob
import json
import time
from pathlib import Path
import subprocess
import lca_bruteforce_params_def
from improvelib.initializer.config import Config

def save_log(dir, result):
    result_file_name_stdout = dir / 'logs.txt'
    if dir.exists() is False: 
        os.makedirs(dir, exist_ok=True)
    with open(result_file_name_stdout, 'w') as file:
        file.write(result.stdout)

def save_time(dir, start):
    time_diff = time.time() - start
    hours = int(time_diff // 3600)
    minutes = int((time_diff % 3600) // 60)
    seconds = time_diff % 60
    time_diff_dict = {'hours': hours,
                    'minutes': minutes,
                    'seconds': seconds}
    with open(Path(dir) / 'runtime.json', 'w') as json_file:
        json.dump(time_diff_dict, json_file, indent=4)



# parameters
filepath = Path(__file__).resolve().parent
cfg = Config() 
params = cfg.initialize_parameters(
    section="LCA",
    pathToModelDir=filepath,
    default_config="lca_bruteforce_params.ini",
    additional_definitions=lca_bruteforce_params_def.additional_definitions
)

output_dir = Path(params['output_dir'])
if output_dir.exists() is False:
    os.makedirs(output_dir, exist_ok=True)


#Model scripts - this should be fine
preprocess_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_preprocess_improve.py")
train_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
infer_python_script = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_infer_improve.py")
print("Created script names.")


# Specify dirs - need to fix this
#y_col_name = params['y_col_name']
MAIN_ML_DATA_DIR = output_dir / 'ml_data' # output_dir_pp, input_dir_train, input_dir_infer
MAIN_MODEL_DIR = output_dir / 'models' # output_dir_train, input_dir_infer
MAIN_INFER_DIR = output_dir / 'infer' # output_dir infer
#MAIN_LOG_DIR = output_dir / 'logs'
#frm.create_outdir(MAIN_LOG_DIR)
print("Created directory names.")
print("output_dir:  ", output_dir)
print("MAIN_ML_DATA_DIR: ", MAIN_ML_DATA_DIR)
print("MAIN_MODEL_DIR:   ", MAIN_MODEL_DIR)
print("MAIN_INFER_DIR:   ", MAIN_INFER_DIR)
#print("MAIN_LOG_DIR:     ", MAIN_LOG_DIR)
# Note! Here input_dir is the location of benchmark data
#splits_dir = Path(params['input_dir']) / params['splits_dir']
#print("Created splits path.")
#print("splits_dir: ", splits_dir)


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


# get splits

lca_split_paths = list(Path(params['lca_splits_dir']).glob(f"{params['dataset']}_split_{params['split_num']}_sz_*.txt"))
lca_split_files = [os.path.basename(x) for x in lca_split_paths]
print(lca_split_files)
print("length of lca: ", len(lca_split_files))
#split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
#split_nums = sorted(set(split_nums))

val_split_file = f"{params['dataset']}_split_{params['split_num']}_val.txt"
test_split_file = f"{params['dataset']}_split_{params['split_num']}_test.txt"

# preprocess
for lca in lca_split_files:
    lca_train_path = params['lca_splits_dir'] + '/' + lca
    print(f"Running IMPROVE scripts with {lca} for training...")
    ### PREPROCESS
    ml_data_dir = MAIN_ML_DATA_DIR / lca.split('.')[0]
    preprocess_start = time.time()
    preprocess_run = ["python", preprocess_python_script,
            "--train_split_file", str(lca_train_path),
            "--val_split_file", str(val_split_file),
            "--test_split_file", str(test_split_file),
            "--input_dir", params['input_dir'], # str("./csa_data/raw_data"),
            "--output_dir", str(ml_data_dir),
            "--y_col_name", str(params['y_col_name']),
            "--input_supp_data_dir", str(supp_data_dir)
    ]
    preprocess_result = subprocess.run(preprocess_run,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            
    # Log and Time
    print(f"preprocess returncode = {preprocess_result.returncode}")
    save_log(ml_data_dir, preprocess_result)
    save_time(ml_data_dir, preprocess_start)

    ### TRAIN
    train_start = time.time()
    model_dir = MAIN_MODEL_DIR / lca.split('.')[0]
    if params["uses_cuda_name"]:
        train_run = ["python", train_python_script,
            "--input_dir", str(ml_data_dir),
            "--output_dir", str(model_dir),
            "--epochs", str(params["epochs"]),
            "--cuda_name", params["cuda_name"],
            "--y_col_name", str(params['y_col_name'])
        ]
    else:
        train_run = ["python", train_python_script,
            "--input_dir", str(ml_data_dir),
            "--output_dir", str(model_dir),
            "--epochs", str(params["epochs"]),
            "--y_col_name", str(params['y_col_name'])
        ]
    train_result = subprocess.run(train_run,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)
    # Log and Time
    print(f"train returncode = {train_result.returncode}")
    save_log(model_dir, train_result)
    save_time(model_dir, train_start)

    ### INFER
    infer_start = time.time()
    infer_dir = MAIN_INFER_DIR / lca.split('.')[0]
    if params["uses_cuda_name"]:
        infer_run = ["python", infer_python_script,
            "--input_data_dir", str(ml_data_dir),
            "--input_model_dir", str(model_dir),
            "--output_dir", str(infer_dir),
            "--cuda_name", params["cuda_name"],
            "--y_col_name", str(params['y_col_name']),
            "--calc_infer_scores", "true"
        ]
    else:
        infer_run = ["python", infer_python_script,
            "--input_data_dir", str(ml_data_dir),
            "--input_model_dir", str(model_dir),
            "--output_dir", str(infer_dir),
            "--y_col_name", str(params['y_col_name']),
            "--calc_infer_scores", "true"
        ]
    infer_result = subprocess.run(infer_run,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)
    # Log and Time
    print(f"infer returncode = {infer_result.returncode}")
    save_log(infer_dir, infer_result)
    save_time(infer_dir, infer_start)
    print(f"Finished IMPROVE scripts with {lca} for training.")

print(f"Finished LCA. Results are in {params['output_dir']}")

 


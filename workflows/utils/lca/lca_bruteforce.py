import os
from pathlib import Path
import lca_bruteforce_params_def
from improvelib.initializer.config import Config




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


# get splits

lca_split_files = list((params['lca_splits_dir']).glob(f"{params['dataset']}_split_{params['split_num']}_sz_*.txt"))
print(lca_split_files)
print("length of lca: ", len(lca_split_files))
#split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
#split_nums = sorted(set(split_nums))

val_split_file = f"{params['dataset']}_split_{params['split_num']}_val.txt"
test_split_file = f"{params['dataset']}_split_{params['split_num']}_test.txt"

# preprocess
for lca in lca_split_files:
    lca_train_path = params['lca_splits_dir'] + '/' + lca
    ml_data_dir = MAIN_ML_DATA_DIR  + "/" + lca.split('.')[0]
    preprocess_run = ["python", preprocess_python_script,
            "--train_split_file", str(lca_train_path),
            "--val_split_file", str(val_split_file),
            "--test_split_file", str(test_split_file),
            "--input_dir", params['input_dir'], # str("./csa_data/raw_data"),
            "--output_dir", str(ml_data_dir),
            "--y_col_name", str(params['y_col_name']),
            "--input_supp_data_dir", str(supp_data_dir)
    ]
    print(preprocess_run)
# train

# inference
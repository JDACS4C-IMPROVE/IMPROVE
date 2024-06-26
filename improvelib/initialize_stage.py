#from improvelib.drp_param_def import *
from improvelib.benchmark_def import Benchmark
from improvelib.parsing_utils import finalize_parameters
import importlib
import argparse

from .helper_utils import str2bool


improve_basic_params = [
    {"name": "benchmark_data_dir",
     "type": str,
     "default": "benchmark_data",
     "help": "Data dir name that stores the benchmark dataset."
    },
    {"name": "raw_data_dir",
     "type": str,
     "default": "raw_data",
     "help": "Data dir name that stores the raw data, including x data, y data, and splits."
    },
    {"name": "x_data_dir",
     "type": str,
     "default": "x_data",
     "help": "Dir name that contains the files with features data (x data)."
    },
    {"name": "y_data_dir",
     "type": str,
     "default": "y_data",
     "help": "Dir name that contains the files with target data (y data)."
    },
    {"name": "splits_dir",
     "type": str,
     "default": "splits",
     "help": "Dir name that contains files that store split ids of the y data file."
    },
    # ---------------------------------------
    {"name": "pred_col_name_suffix",
     "type": str,
     "default": "_pred",
     "help": "Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
    },
    {
        "name": "config_file",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "specify model configuration file",
    },
    {"name": "y_col_name", # workflow
     "type": str,
     "default": "auc",
     "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
    },
    {"name": "y_data_suffix", # default # TODO. rename y_data_stage_fname_suffix?
      "type": str,
      "default": "y_data",
      "help": "Suffix to compose file name for storing true y dataframe."
    },
    {
        "name": "verbose",
        "abv": "v",
        "type": str2bool,
        "default": False,
        "help": "increase output verbosity.",
    },
    {"name": "logfile", "abv": "l", "type": str, "default": None, "help": "log file"},
   {
        "name": "task",
        "type": str,
        "default": "regression",
        "choices": ["regression", "classification"],
        "help": "Prediction task type.",
    },

]

improve_preprocess_params = [
    {
        "name": "improvespecificpreprocessthing",
        "type": str,
        "default": "ABC",
        "help": "whatever.",
    },
    {"name": "train_split_file", # workflow
     "default": "train_split.txt",
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., \
             'split_0_train_id', 'split_0_train_size_1024').",
    },
    {"name": "val_split_file", # workflow
     "default": "val_split.txt",
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_val_id').",
    },
    {"name": "test_split_file", # workflow
     "default": "test_split.txt",
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_test_id').",
    },
    {"name": "ml_data_outdir", # workflow # TODO. this was previously ml_data_outpath
     "type": str,
     "default": "./ml_data",
     "help": "Path to save ML data (data files that can be fet to the prediction model).",
    },
    {"name": "data_format",  # [Req] Must be specified for the model! TODO. rename to ml_data_format?
      "type": str,
      "default": "",
      "help": "File format to save the ML data file (e.g., '.pt', '.tfrecords')",
    },
]
improve_train_params = [
    {
        "name": "improvespecificTRAINthing",
        "type": str,
        "default": "ABC",
        "help": "whatever.",
    },
    {"name": "train_ml_data_dir", # workflow
     # "action": "store",
     "type": str,
     "default": "./ml_data",
     "help": "Datadir where train data is stored."
    },
    {"name": "val_ml_data_dir", # workflow
     # "action": "store",
     "type": str,
     "default": "./ml_data",
     "help": "Datadir where val data is stored."
    },
    {"name": "model_outdir", # workflow
     "type": str,
     "default": "./out_model", # csa_data/models/
     "help": "Dir to save trained models.",
    },
    # {"name": "model_params",
    {"name": "model_file_name",  # default # TODO: this was previously model_file_name
     "type": str,
     # "default": "model.pt",
     "default": "model",
     "help": "Filename to store trained model (str is w/o file_format)."
    },
    {"name": "model_file_format",  # [Req]
     "type": str,
     "default": ".pt",
     "help": "File format to save the trained model."
    },
    {"name": "epochs", # [Req]
     "type": int,
     "default": 20,
     "help": "Training epochs."
    },
    {"name": "batch_size", # [Req]
     "type": int,
     "default": 64,
     "help": "Trainig batch size."
    },
    {"name": "val_batch", # [Req]
     "type": int,
     "default": 64,
     # "default": argparse.SUPPRESS,
     "help": "Validation batch size."
    },
    {"name": "loss", # [Req] used in compute_metrics
     "type": str,
     "default": "mse",
     "help": "Loss metric."
    },
    {"name": "learning_rate",
      "type": float,
      "default": 0.0001,
      "help": "Learning rate for the optimizer."
    },
    {"name": "early_stop_metric", # [Req]
     "type": str,
     "default": "mse",
     "help": "Prediction performance metric to monitor for early stopping during \
             model training (e.g., 'mse', 'rmse').",
    },
    {"name": "patience", # [Req]
     "type": int,
     "default": 20,
     # "default": argparse.SUPPRESS,
     "help": "Iterations to wait for validation metrics getting worse before \
             stopping training.",
    },
    {"name": "y_data_preds_suffix", # default
     "type": str,
     "default": "predicted",
     "help": "Suffix to use for name of file to store inference results."
    },
    {"name": "json_scores_suffix", # default
     "type": str,
     "default": "scores",
     "help": "Suffix to use for name of file to store scores."
    },
]

improve_infer_params = [
    {
        "name": "improvespecificINFERthing",
        "type": str,
        "default": "ABC",
        "help": "whatever.",
    },
    {"name": "test_ml_data_dir", # workflow
     # "action": "store",
     "type": str,
     "default": "./ml_data",
     "help": "Datadir where test data is stored."
    },
    {"name": "model_dir", # workflow
     "type": str,
     "default": "./out_model", # csa_data/models/
     "help": "Dir to save inference results.",
    },
    {"name": "infer_outdir", # workflow
     "type": str,
     "default": "./out_infer", # csa_data/infer/
     "help": "Dir to save inference results.",
    },
    # {"name": "test_data_processed",  # TODO: is this test_data.pt?
    #  "action": "store",
    #  "type": str,
    #  "help": "Name of pytorch processed test data file."
    # },
    {"name": "test_batch",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Test batch size.",
    },
    {"name": "y_data_preds_suffix", # shared with train
     "type": str,
     "default": "predicted",
     "help": "Suffix to use for name of file to store inference results."
    },
    {"name": "json_scores_suffix", # shared with train
     "type": str,
     "default": "scores",
     "help": "Suffix to use for name of file to store scores."
    },
    {"name": "loss", # shared with train
     "type": str,
     "default": "mse",
     "help": "Loss metric."
    },
]

def initialize_benchmark(filepath, defmodel, additional_definitions):
     bmk = Benchmark(
            filepath=filepath,
            defmodel=defmodel,
            framework="pytorch",
            prog="frm",
            desc="Framework functionality in IMPROVE",
            additional_definitions=additional_definitions
        ) 
     return bmk 

def initialize_params(bmk):
    params = finalize_parameters(bmk)
    return params
 
 
def get_required_params(req, defs):
    if req is None or len(req) == 0:
        return defs
    else:
        r = set(req)
        defs = [d for d in defs if d['name'] in r]
        return defs

def determine_stage_specific_params(filename, filepath, application, defmodel, model_preprocess_params, model_train_params, model_infer_params, required):

    #model_param_def_fname = filepath + "/" + application + "_param_def"
    # get app specific params
    app_param_def_fname = "." + application + "_param_def"
    app_params = importlib.import_module(app_param_def_fname, 'improvelib')
    if application == "drp":
        data_params = importlib.import_module(".drug_resp_pred", 'improvelib').drp_data_params
    #to do -- error handing
    if "preprocess" in filename:
        data_params = get_required_params(required, data_params)
        additional_definitions = improve_preprocess_params + app_params.app_preprocess_params + model_preprocess_params + improve_basic_params + data_params
        benchmark = initialize_benchmark(filepath, defmodel, additional_definitions)
        params = initialize_params(benchmark)
    elif "train" in filename:
        global improve_train_params
        improve_train_params = get_required_params(required, improve_train_params)
        additional_definitions = improve_train_params + app_params.app_train_params + model_train_params + improve_basic_params
        benchmark = initialize_benchmark(filepath, defmodel, additional_definitions)
        params = initialize_params(benchmark)
    elif "infer" in filename:
        additional_definitions = improve_infer_params + app_params.app_infer_params + model_infer_params + improve_basic_params
        benchmark = initialize_benchmark(filepath, defmodel, additional_definitions)
        params = initialize_params(benchmark)
    else:
        print("File name does not contain 'preprocess', 'train', or 'infer'!")
    return benchmark, params

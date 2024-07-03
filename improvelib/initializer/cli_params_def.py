"""
TODO
Do we need all 4 lists of parameters?
Determine which parameters we really need in CLI
Do we want to rename any parameter

"""


# Parameters that are relevant to all IMPROVE models
# Defaults for these args are expected to be used
improve_basic_conf = [
    {"name": "log_level",
     "type": str,
     "default": "DEBUG",
     "help": "Logger verbosity"
     },
    {"name": "input_dir",
     "type": str,
     "default": "./",
     "help": "Data dir name that stores the raw data, including x data, y data, and splits."
    },
    {"name": "output_dir",
     "type": str,
     "default": "./",
     "help": "Data dir name that stores the raw data, including x data, y data, and splits."
    },
    {"name": "config_file",
     "type": str,
     "default": None,
     "help": "Configuration file for the script"
     },

]

# Parameters that are relevant to all IMPROVE preprocessing scripts
improve_preprocess_conf = [
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
    # Values for these args are expected to be passed:
    # train_split_file, val_split_file, test_split_file
    {"name": "train_split_file",  # workflow
     "default": None,
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., \
             'split_0_train_id', 'split_0_train_size_1024').",
     },
    {"name": "val_split_file",  # workflow
     "default": None,
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_val_id').",
     },
    {"name": "test_split_file",  # workflow
     "default": None,
     "type": str,
     # "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_test_id').",
     },
    # ---------------------------------------
    {"name": "ml_data_outdir",  # workflow # TODO. this was previously ml_data_outpath
     "type": str,
     "default": "./ml_data",
     "help": "Path to save ML data (data files that can be fet to the prediction model).",
     },
    {"name": "data_format",  # [Req] Must be specified for the model! TODO. rename to ml_data_format?
     "type": str,
     "default": ".parquet",
     "help": "File format to save the ML data file (e.g., '.pt', '.tfrecords')",
     },
    {"name": "y_col_name",  # workflow
     "type": str,
     "default": "auc",
     "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
     },
    # ---------------------------------------
    # {"name": "x_data_suffix",  # TODO. rename x_data_stage_fname_suffix?
    #   "type": str,
    #   "default": "data",
    #   "help": "Suffix to compose file name for storing x data (e.g., ...)."
    # },
    {"name": "y_data_suffix",  # default # TODO. rename y_data_stage_fname_suffix?
     "type": str,
     "default": "y_data",
     "help": "Suffix to compose file name for storing true y dataframe."
     },

]

# Parameters that are relevant to all IMPROVE training scripts
improve_train_conf = [
    {"name": "train_ml_data_dir",  # workflow
     # "action": "store",
     "type": str,
     "default": "./ml_data",
     "help": "Datadir where train data is stored."
     },
    {"name": "val_ml_data_dir",  # workflow
     # "action": "store",
     "type": str,
     "default": "./ml_data",
     "help": "Datadir where val data is stored."
     },
    {"name": "model_outdir",  # workflow
     "type": str,
     "default": "./out_model",  # csa_data/models/
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
    # ---------------------------------------
    {"name": "epochs",  # [Req]
     "type": int,
     "default": 20,
     "help": "Training epochs."
     },
    {"name": "batch_size",  # [Req]
     "type": int,
     "default": 64,
     "help": "Trainig batch size."
     },
    {"name": "val_batch",  # [Req]
     "type": int,
     "default": 64,
     # "default": argparse.SUPPRESS,
     "help": "Validation batch size."
     },
    {"name": "loss",  # [Req] used in compute_metrics
     "type": str,
     "default": "mse",
     "help": "Loss metric."
     },
    # {"name": "optimizer",
    #  "type": str,
    #  "default": "adam",
    #  "help": "Optimizer for backpropagation."
    # },
    # {"name": "learning_rate",
    #  "type": float,
    #  "default": 0.0001,
    #  "help": "Learning rate for the optimizer."
    # },
    # {"name": "train_data_processed",
    #  "action": "store",
    #  "type": str,
    #  "help": "Name of pytorch processed train data file."
    # },
    # {"name": "val_data_processed",
    #  "action": "store",
    #  "type": str,
    #  "help": "Name of pytorch processed val data file."
    # },
    # {"name": "model_eval_suffix", # TODO: what's that?
    # y_data_stage_preds_fname_suffix
    {"name": "early_stop_metric",  # [Req]
     "type": str,
     "default": "mse",
     "help": "Prediction performance metric to monitor for early stopping during \
             model training (e.g., 'mse', 'rmse').",
     },
    {"name": "patience",  # [Req]
     "type": int,
     "default": 20,
     # "default": argparse.SUPPRESS,
     "help": "Iterations to wait for validation metrics getting worse before \
             stopping training.",
     },
    {"name": "y_data_preds_suffix",  # default
     "type": str,
     "default": "predicted",
     "help": "Suffix to use for name of file to store inference results."
     },
    {"name": "json_scores_suffix",  # default
     "type": str,
     "default": "scores",
     "help": "Suffix to use for name of file to store scores."
     },
    {"name": "pred_col_name_suffix", # also defined in improve_infer_conf
     "type": str,
     "default": "_pred",
     "help": "Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
     },

]

# Parameters that are relevant to all IMPROVE testing scripts
improve_infer_conf = [
    {"name": "test_ml_data_dir",  # workflow
     # "action": "store",
     "type": str,
     "default": "./ml_data",
     "help": "Datadir where test data is stored."
     },
    {"name": "model_dir",  # workflow
     "type": str,
     "default": "./out_model",  # csa_data/models/
     "help": "Dir to save inference results.",
     },
    {"name": "infer_outdir",  # workflow
     "type": str,
     "default": "./out_infer",  # csa_data/infer/
     "help": "Dir to save inference results.",
     },
    # {"name": "test_data_processed",  # TODO: is this test_data.pt?
    #  "action": "store",
    #  "type": str,
    #  "help": "Name of pytorch processed test data file."
    # },
    {"name": "test_batch",
     "type": int,
     "default": 64,
     "help": "Test batch size.",
     },
    {"name": "pred_col_name_suffix", # also defined in improve_train_conf
     "type": str,
     "default": "_pred",
     "help": "Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
     },

]


# Combine improve configuration into additional_definitions
cli_param_definitions = improve_basic_conf + \
    improve_preprocess_conf + \
    improve_train_conf + \
    improve_infer_conf

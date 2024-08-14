"""
The help section strings of certain parameters is prepended with the following brackets,
encoding a suggested deprecation.
"[Dep+]" -- parameters we should try to deprecate in the upcoming release
"[Dep?]" -- parameters we might want to deprecate in the current or future release
"[Dep]"  -- parameters we plan to keep in the upcoming release, but deprecate in the future
"""

from improvelib.utils import StoreIfPresent


# Parameters relevant to all IMPROVE model models
# These parameters will be accessible in all model scripts (preprocess, train, infer)
improve_basic_conf = [
    {"name": "log_level",
     "type": str,
     "default": "DEBUG",
     "help": "Logger verbosity. Options: ERROR, etc etc"
    },
    {"name": "input_dir",
     "type": str,
     "default": "./",
     "help": "Dir containing input data for a given model script. The content \
             will depend on the model script (preprocess, train, infer)."
    },
    {"name": "output_dir",
     "type": str,
     "default": "./",
     "help": "Dir where the outputs of a given model script will be saved. The \
             content will depend on the model script (preprocess, train, infer)."
    },
    {"name": "config_file",
     "type": str,
     "default": None,
     "help": "Configuration file for the model. The parameters defined in the \
             file override the default parameter values."
    },
    {"name": "param_log_file",
     "type": str,
     "default": "param_log_file.txt",
     "help": "Log of final parameters used for run. Saved in out_dir if file name, can be an absolute path."
    },
    # ---------------------------------------
    {"name": "data_format",  # [Req] depends on the DL framework that used by the model
     "type": str,
     "default": ".parquet", # Note! this default assumes LGBM model
     # "required": True, # TODO if this is required param, then remove default
     "help": "File format to save the ML data file (e.g., '.pt', '.tfrecords')",
    },
    # ---------------------------------------
    # # Explore new args
    # {"name": "ml_data_dir",
    #  "type": str,
    #  # "default": "./",
    #  "action": StoreIfPresent,
    #  "help": "Dir with input data (ML data) for a model (data files that can \
    #          be fed to the prediction model)."
    # },
    # {"name": "model_dir",
    #  "type": str,
    #  # "default": "./",
    #  "action": StoreIfPresent,
    #  "help": "Dir with trained model and other relevant output (i.e., \
    #          val_scores.json)."
    # },
    # {"name": "infer_dir",
    #  "type": str,
    #  # "default": "./",
    #  "action": StoreIfPresent,
    #  "help": "Dir with inference run output."
    # },
]

# Parameters that are relevant to all IMPROVE preprocessing scripts
improve_preprocess_conf = [
    {"name": "x_data_dir", # default expected
     "type": str,
     "default": "x_data",
     "help": "Dir name that contains the files with features data (x data)."
     },
    {"name": "y_data_dir", # default expected
     "type": str,
     "default": "y_data",
     "help": "Dir name that contains the files with target data (y data)."
     },
    {"name": "splits_dir", # default expected
     "type": str,
     "default": "splits",
     "help": "Dir name that contains files that store split ids of the y data file."
     },
    # ---------------------------------------
    {"name": "train_split_file",
     "type": str,
     "default": "fake", # NCK default needed
     "required": True,
     "help": "The path to the file that contains the train split ids (e.g., \
             'split_0_train_id', 'split_0_train_size_1024').",
     },
    {"name": "val_split_file",
     "type": str,
     "default": "fake", # NCK default needed
     "required": True,
     "help": "The path to the file that contains the val split ids (e.g., \
             'split_0_val_id').",
     },
    {"name": "test_split_file",
     "type": str,
     "default": "fake", # NCK default needed
     "required": True,
     "help": "The path to the file that contains the test split ids (e.g., \
             'split_0_test_id').",
     },
    # ---------------------------------------
    {"name": "ml_data_outdir",  # TODO plan using output_dir instead
     "type": str,
     "action": StoreIfPresent,
     "help": "[Dep+] Path to save ML data (data files that can be fet to the prediction model).",
     },

]

# Parameters relevant to all IMPROVE train scripts
improve_train_conf = [
    {"name": "train_ml_data_dir",  # TODO plan using input_dir instead
     "type": str,
     # "default": "./ml_data",
     "action": StoreIfPresent,
     "help": "[Dep+] Datadir where train data is stored."
     },
    {"name": "val_ml_data_dir",  # TODO plan using input_dir instead
     "type": str,
     # "default": "./ml_data",
     "action": StoreIfPresent,
     "help": "[Dep+] Datadir where val data is stored."
     },
    {"name": "model_outdir",  # TODO plan using output_dir instead
     "type": str,
     # "default": "./out_model",  # csa_data/models/
     "action": StoreIfPresent,
     "help": "[Dep+] Dir to save trained models.",
     },
    # ---------------------------------------
    {"name": "model_file_name",  # default expected
     "type": str,
     "default": "model",
     "help": "[Dep?] Filename to store trained model (str is w/o file_format)."
     },
    {"name": "model_file_format",  # [Req] depends on the DL framework used by the model
     "type": str,
     "default": ".pt", # Note! this default assumes PyTorch model
     # "required": True, # TODO if this is required param, then remove default
     "help": "[Dep?] File format to save the trained model."
     },
    # ---------------------------------------
    {"name": "epochs",
     "type": int,
     "default": 7, # NCK default needed
     "required": True,
     "help": "Training epochs."
     },
    {"name": "learning_rate",
     "type": float,
     "default": 7, # NCK default needed
     "required": True,
     "help": "Learning rate for the optimizer."
     },     
    {"name": "batch_size",
     "type": int,
     "default": 7, # NCK default needed
     "required": True,
     "help": "Trainig batch size."
     },
    {"name": "val_batch",
     "type": int,
     "default": 64,
     "help": "Validation batch size."
     },
    # ---------------------------------------
    {"name": "loss",  # TODO used in compute_metrics(), but probably can be removed
     "type": str,
     "default": "mse",
     "help": "[Dep?] Loss metric."
     },
    {"name": "early_stop_metric",  # [Req] TODO consider moving to app or model (with patience)
     "type": str,
     "default": "mse",
     "help": "Prediction performance metric to monitor for early stopping during \
             model training (e.g., 'mse', 'rmse').",
     },
    {"name": "patience",  # [Req] TODO consider moving to app or model params (with early_stop_metric)
     "type": int,
     "default": 20,
     "help": "Iterations to wait for a validation metric to get worse before \
             stop training.",
     },
    # ---------------------------------------
    # TODO y_data_preds_suffix, json_scores_suffix, pred_col_name_suffix
    # are currently used in utils.py (previously framework.py)
    # We plan to hard-code these in the future release and deprecate.
    # Defined in improve_train_conf and improve_infer_conf
    {"name": "y_data_preds_suffix", # default expected
     "type": str,
     "default": "predicted",
     "help": "[Dep] Suffix to use for file name that stores predictions."
     },
    {"name": "json_scores_suffix", # default expected
     "type": str,
     "default": "scores",
     "help": "[Dep] Suffix to use for file name that stores scores."
     },
    {"name": "pred_col_name_suffix", # default expected
     "type": str,
     "default": "_pred",
     "help": "[Dep] Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
     },

]

# Parameters relevant to all IMPROVE infer scripts
improve_infer_conf = [
    {"name": "test_ml_data_dir", # TODO plan using input_dir instead
     "type": str,
     # "default": "./ml_data",
     "action": StoreIfPresent,
     "help": "[Dep+] Datadir where test data is stored."
     },
    {"name": "model_dir", # TODO plan using input_dir instead
     "type": str,
     # "default": "./out_model",  # csa_data/models/
     "action": StoreIfPresent,
     "help": "[Dep+] Dir to save inference results.",
     },
    {"name": "infer_outdir",  # TODO plan using output_dir instead
     "type": str,
     # "default": "./out_infer",
     "action": StoreIfPresent,
     "help": "[Dep+] Dir to save inference results.",
     },
    # ---------------------------------------
    {"name": "test_batch", # TODO should this be infer_batch?
     "type": int,
     "default": 64,
     "help": "Test batch size.",
     },
    # ---------------------------------------
    # TODO. y_data_preds_suffix, json_scores_suffix, pred_col_name_suffix
    # are currently used in utils.py (previously framework.py)
    # We plan to hard-code these in the future release and deprecate.
    # Defined in improve_train_conf and improve_infer_conf
    {"name": "y_data_preds_suffix", # default expected
     "type": str,
     "default": "predicted",
     "help": "[Dep] Suffix to use for file name that stores predictions."
     },
    {"name": "json_scores_suffix", # default expected
     "type": str,
     "default": "scores",
     "help": "[Dep] Suffix to use for file name that stores scores."
     },
    {"name": "pred_col_name_suffix", # default expected
     "type": str,
     "default": "_pred",
     "help": "[Dep] Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
     },
     # --------------------------------------- NCK: infer is looking for these too
    {"name": "model_file_name",  # default expected
     "type": str,
     "default": "model",
     "help": "[Dep?] Filename to store trained model (str is w/o file_format)."
     },
    {"name": "model_file_format",  # [Req] depends on the DL framework used by the model
     "type": str,
     "default": ".pt", # Note! this default assumes PyTorch model
     # "required": True, # TODO if this is required param, then remove default
     "help": "[Dep?] File format to save the trained model."
     },
     {"name": "loss",  
     "type": str,
     "default": "mse",
     "help": "[Dep?] Loss metric."
     },
    # ---------------------------------------
    {"name": "input_data_dir",
     "type": str,
     "action": StoreIfPresent,
     "help": "Dir where data for inference is stored."
     },
    {"name": "input_model_dir",
     "type": str,
     "action": StoreIfPresent,
     "help": "Dir where model is stored.",
     },

]


# Combine improve param definition into additional_definitions
cli_param_definitions = improve_basic_conf + \
                        improve_preprocess_conf + \
                        improve_train_conf + \
                        improve_infer_conf

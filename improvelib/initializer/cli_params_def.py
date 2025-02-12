"""Parameter configurations for the IMPROVE model stages.

This module defines parameter configurations for preprocessing, training, and inference
stages of the IMPROVE model. It includes both common parameters shared across all stages
and stage-specific parameters.

Variables:
    improve_basic_conf (list): Common parameters for all IMPROVE model stages.
    improve_preprocess_conf (list): Parameters specific to preprocessing stage.
    improve_train_conf (list): Parameters specific to training stage.
    improve_infer_conf (list): Parameters specific to inference stage.
    cli_param_definitions (list): Combined list of all parameter definitions.
"""

import argparse

from improvelib.utils import StoreIfPresent, str2bool

# Parameters relevant to all IMPROVE model models
# These parameters will be accessible in all model scripts (preprocess, train, infer)
improve_basic_conf = [
    {
        "name": "log_level",
        "type": str,
        "default": "DEBUG",
        "help": (
            "Logger verbosity level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        ),
    },
    {
        "name": "input_dir",
        "type": str,
        "default": "./",
        "help": (
            "Directory containing input data for a given model script. The content "
            "will depend on the model script (preprocess, train, infer)."
        ),
    },
    {
        "name": "output_dir",
        "type": str,
        "default": "./",
        "help": (
            "Directory where the outputs of a given model script will be saved. The "
            "content will depend on the model script (preprocess, train, infer)."
        ),
    },
    {
        "name": "config_file",
        "type": str,
        "default": None,
        "help": (
            "Configuration file for the model. The parameters defined in the "
            "file override the default parameter values."
        ),
    },
    {
        "name": "param_log_file",
        "type": str,
        "default": "param_log_file.txt",
        "help": (
            "Log of final parameters used for run. Saved in out_dir if file name, "
            "can be an absolute path."
        ),
    },
    {
        "name": "data_format",
        "type": str,
        "default": ".parquet",
        "help": "File format to save the ML data file (e.g., '.pt', '.tfrecords').",
    },
    {
        "name": "input_supp_data_dir",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": (
            "Directory containing supplementary data in addition to benchmark data "
            "(usually model-specific data)."
        ),
    },
]

# Parameters that are relevant to all IMPROVE preprocessing scripts
improve_preprocess_conf = [
    {
        "name": "x_data_dir",
        "type": str,
        "default": "x_data",
        "help": "Directory name that contains the files with features data (x data).",
    },
    {
        "name": "y_data_dir",
        "type": str,
        "default": "y_data",
        "help": "Directory name that contains the files with target data (y data).",
    },
    {
        "name": "splits_dir",
        "type": str,
        "default": "splits",
        "help": (
            "Directory name that contains files that store split ids of the y data file."
        ),
    },
    {
        "name": "train_split_file",
        "type": str,
        "default": "CCLE_split_0_train.txt",
        "required": True,
        "help": (
            "The path to the file that contains the train split indices or ids "
            "(e.g., 'CCLE_split_0_train.txt')."
        ),
    },
    {
        "name": "val_split_file",
        "type": str,
        "default": "CCLE_split_0_val.txt",
        "required": True,
        "help": (
            "The path to the file that contains the validation split indices or ids "
            "(e.g., 'CCLE_split_0_val.txt')."
        ),
    },
    {
        "name": "test_split_file",
        "type": str,
        "default": "CCLE_split_0_test.txt",
        "required": True,
        "help": (
            "The path to the file that contains the test split indices or ids "
            "(e.g., 'CCLE_split_0_test.txt')."
        ),
    },
]

# Parameters relevant to all IMPROVE train scripts
improve_train_conf = [
    {
        "name": "model_file_name",
        "type": str,
        "default": "model",
        "help": "File name (without extension) used for saving or loading the trained model.",
    },
    {
        "name": "model_file_format",
        "type": str,
        "default": ".pt",
        "help": "File extension used for saving or loading the trained model.",
    },
    {
        "name": "epochs",
        "type": int,
        "default": 7,
        "required": True,
        "help": "Number of training epochs.",
    },
    {
        "name": "learning_rate",
        "type": float,
        "default": 7,
        "required": True,
        "help": "Learning rate for the optimizer.",
    },
    {
        "name": "batch_size",
        "type": int,
        "default": 7,
        "required": True,
        "help": "Training batch size.",
    },
    {
        "name": "val_batch",
        "type": int,
        "default": 64,
        "help": "Validation batch size.",
    },
    {
        "name": "loss",
        "type": str,
        "default": "mse",
        "help": "Loss metric.",
    },
    {
        "name": "early_stop_metric",
        "type": str,
        "default": "mse",
        "help": (
            "Prediction performance metric to monitor for early stopping during "
            "model training (e.g., 'mse', 'rmse')."
        ),
    },
    {
        "name": "patience",
        "type": int,
        "default": 20,
        "help": (
            "Number of iterations to wait for improvement in validation metric "
            "before stopping training."
        ),
    },
    {
        "name": "metric_type",
        "type": str,
        "default": "regression",
        "help": (
            "Metrics appropriate for the given task. Options are 'regression' "
            "or 'classification'."
        ),
    },
]

# Parameters relevant to all IMPROVE infer scripts
improve_infer_conf = [
    {
        "name": "infer_batch",
        "type": int,
        "default": 64,
        "help": "Inference batch size.",
    },
    {
        "name": "model_file_name",
        "type": str,
        "default": "model",
        "help": "File name (without extension) used for saving or loading the trained model.",
    },
    {
        "name": "model_file_format",
        "type": str,
        "default": ".pt",
        "help": "File extension used for saving or loading the trained model.",
    },
    {
        "name": "loss",
        "type": str,
        "default": "mse",
        "help": "Loss metric.",
    },
    {
        "name": "input_data_dir",
        "type": str,
        "default": "./",
        "help": "Directory where data for inference is stored.",
    },
    {
        "name": "input_model_dir",
        "type": str,
        "default": "./",
        "help": "Directory where the model is stored.",
    },
    {
        "name": "calc_infer_scores",
        "type": str2bool,
        "default": False,
        "help": "Optional. Calculate scores during inference.",
    },
    {
        "name": "metric_type",
        "type": str,
        "default": "regression",
        "help": (
            "Metrics appropriate for the given task. Options are 'regression' "
            "or 'classification'."
        ),
    },
]

# Combine all parameter definitions from different stages into a single list
# called cli_param_definitions for unified access.
cli_param_definitions = (
    improve_basic_conf +
    improve_preprocess_conf +
    improve_train_conf +
    improve_infer_conf
)

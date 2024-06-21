basic_conf = [
    {
        "name": "config_file",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "specify model configuration file",
    },
    {
        "name": "data_type",
        "abv": "d",
        "type": str,
        "default": argparse.SUPPRESS,
        "choices": ["f16", "f32", "f64"],
        "help": "default floating point.",
    },
    {
        "name": "rng_seed",
        "abv": "r",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "random number generator seed.",
    },
    {"name": "train_bool", "type": str2bool, "default": True, "help": "train model."},
    {
        "name": "eval_bool",
        "type": str2bool,
        "default": argparse.SUPPRESS,
        "help": "evaluate model (use it for inference).",
    },
    {
        "name": "timeout",
        "action": "store",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "seconds allowed to train model (default: no timeout).",
    },
    {
        "name": "gpus",
        "nargs": "+",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "set IDs of GPUs to use.",
    },
    {
        "name": "profiling",
        "abv": "p",
        "type": str2bool,
        "default": False,
        "help": "Turn profiling on or off.",
    },
]

input_output_conf = [
    {
        "name": "save_path",
        "abv": "s",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "file path to save model snapshots.",
    },
    {
        "name": "model_name",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "specify model name to use when building filenames for saving.",
    },
    {
        "name": "home_dir",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "set home directory.",
    },
    {
        "name": "train_data",
        "action": "store",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "training data filename.",
    },
    {
        "name": "val_data",
        "action": "store",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "validation data filename.",
    },
    {
        "name": "test_data",
        "type": str,
        "action": "store",
        "default": argparse.SUPPRESS,
        "help": "testing data filename.",
    },
    {
        "name": "output_dir",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "output directory.",
    },
    {
        "name": "data_url",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "set data source url.",
    },
    {
        "name": "experiment_id",
        "type": str,
        "default": "EXP000",
        "help": "set the experiment unique identifier.",
    },
    {
        "name": "run_id",
        "type": str,
        "default": "RUN000",
        "help": "set the run unique identifier.",
    },
]

logging_conf = [
    {
        "name": "verbose",
        "abv": "v",
        "type": str2bool,
        "default": False,
        "help": "increase output verbosity.",
    },
    {"name": "logfile", "abv": "l", "type": str, "default": None, "help": "log file"},
]

data_preprocess_conf = [
    {
        "name": "scaling",
        "type": str,
        "default": argparse.SUPPRESS,
        "choices": ["minabs", "minmax", "std", "none"],
        "help": "type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization.",
    },
    {
        "name": "shuffle",
        "type": str2bool,
        "default": False,
        "help": "randomly shuffle data set (produces different training and testing partitions each run depending on the seed)",
    },
    {
        "name": "feature_subsample",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features",
    },
]

extra_conf = [
    {
        "name": "jupyter",
        "abv": "f",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "Reserve abv f for Jupyter notebook",
    },
    {
        "name": "HistoryManager.hist_file=:memory",
        "abv": "HistoryManager.hist_file=:memory",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "Reserve abv f for Jupyter notebook",
    },
]

model_conf = [
    {
        "name": "dense",
        "nargs": "+",
        "type": int,
        "help": "number of units in fully connected layers in an integer array.",
    },
    {
        "name": "conv",
        "nargs": "+",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "integer array describing convolution layers: conv1_filters, conv1_filter_len, conv1_stride, conv2_filters, conv2_filter_len, conv2_stride ....",
    },
    {
        "name": "locally_connected",
        "type": str2bool,
        "default": argparse.SUPPRESS,
        "help": "use locally connected layers instead of convolution layers.",
    },
    {
        "name": "activation",
        "abv": "a",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "Keras activation function to use in inner layers: relu, tanh, sigmoid...",
    },
    {
        "name": "out_activation",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "Keras activation function to use in out layer: softmax, linear, ...",
    },
    {
        "name": "lstm_size",
        "nargs": "+",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "integer array describing size of LSTM internal state per layer.",
    },
    {
        "name": "recurrent_dropout",
        "action": "store",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "ratio of recurrent dropout.",
    },
    {
        "name": "dropout",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "ratio of dropout used in fully connected layers.",
    },
    {
        "name": "pool",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "pooling layer length.",
    },
    {
        "name": "batch_normalization",
        "type": str2bool,
        "default": argparse.SUPPRESS,
        "help": "use batch normalization.",
    },
    {
        "name": "loss",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "Keras loss function to use: mse, ...",
    },
    {
        "name": "optimizer",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "Keras optimizer to use: sgd, rmsprop, ...",
    },
    {
        "name": "metrics",
        "type": str,
        "default": argparse.SUPPRESS,
        "help": "metrics to evaluate performance: accuracy, ...",
    },
]

training_conf = [
    {
        "name": "epochs",
        "type": int,
        "abv": "e",
        "default": argparse.SUPPRESS,
        "help": "number of training epochs.",
    },
    {
        "name": "batch_size",
        "type": int,
        "abv": "z",
        "default": argparse.SUPPRESS,
        "help": "batch size.",
    },
    {
        "name": "learning_rate",
        "abv": "lr",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "overrides the learning rate for training.",
    },
    {
        "name": "early_stop",
        "type": str2bool,
        "default": argparse.SUPPRESS,
        "help": "activates Keras callback for early stopping of training in function of the monitored variable specified.",
    },
    {
        "name": "momentum",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "overrides the momentum to use in the SGD optimizer when training.",
    },
    {
        "name": "initialization",
        "type": str,
        "default": argparse.SUPPRESS,
        "choices": [
            "constant",
            "uniform",
            "normal",
            "glorot_uniform",
            "glorot_normal",
            "lecun_uniform",
            "he_normal",
        ],
        "help": "type of weight initialization; 'constant': to 0; 'uniform': to [-0.05,0.05], 'normal': mean 0, stddev 0.05; 'glorot_uniform': [-lim,lim] with lim = sqrt(6/(fan_in+fan_out)); 'lecun_uniform' : [-lim,lim] with lim = sqrt(3/fan_in); 'he_normal' : mean 0, stddev sqrt(2/fan_in).",
    },
    {
        "name": "val_split",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "fraction of data to use in validation.",
    },
    {
        "name": "train_steps",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "overrides the number of training batches per epoch if set to nonzero.",
    },
    {
        "name": "val_steps",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "overrides the number of validation batches per epoch if set to nonzero.",
    },
    {
        "name": "test_steps",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "overrides the number of test batches per epoch if set to nonzero.",
    },
    {
        "name": "train_samples",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "overrides the number of training samples if set to nonzero.",
    },
    {
        "name": "val_samples",
        "type": int,
        "default": argparse.SUPPRESS,
        "help": "overrides the number of validation samples if set to nonzero.",
    },
]

cyclic_learning_conf = [
    {
        "name": "clr_flag",
        "type": str2bool,
        "default": argparse.SUPPRESS,
        "help": "CLR flag (boolean).",
    },
    {
        "name": "clr_mode",
        "type": str,
        "default": argparse.SUPPRESS,
        "choices": ["trng1", "trng2", "exp"],
        "help": "CLR mode (default: trng1).",
    },
    {
        "name": "clr_base_lr",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "Base lr for cycle lr.",
    },
    {
        "name": "clr_max_lr",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "Max lr for cycle lr.",
    },
    {
        "name": "clr_gamma",
        "type": float,
        "default": argparse.SUPPRESS,
        "help": "Gamma parameter for learning cycle LR.",
    },
]

ckpt_conf = [
    {
        "name": "ckpt_restart_mode",
        "type": str,
        "default": "auto",
        "choices": ["off", "auto", "required"],
        "help": "Mode to restart from a saved checkpoint file, choices are 'off', 'auto', 'required'.",
    },
    {
        "name": "ckpt_checksum",
        "type": str2bool,
        "default": False,
        "help": "Checksum the restart file after read+write.",
    },
    {
        "name": "ckpt_skip_epochs",
        "type": int,
        "default": 0,
        "help": "Number of epochs to skip before saving epochs.",
    },
    {
        "name": "ckpt_directory",
        "type": str,
        # When unset, the default is handled by the ckpt modules
        "default": None,
        "help": "Base directory in which to save checkpoints.",
    },
    {
        "name": "ckpt_save_best",
        "type": str2bool,
        "default": True,
        "help": "Toggle saving best model.",
    },
    {
        "name": "ckpt_save_best_metric",
        "type": str,
        "default": "val_loss",
        "help": "Metric for determining when to save best model.",
    },
    {
        "name": "ckpt_save_weights_only",
        "type": str2bool,
        "default": False,
        "help": "Toggle saving only weights (not optimizer) (NYI).",
    },
    {
        "name": "ckpt_save_interval",
        "type": int,
        "default": 0,
        "help": "Interval to save checkpoints.",
    },
    {
        "name": "ckpt_keep_mode",
        "type": str,
        "default": "linear",
        "choices": ["linear", "exponential"],
        "help": "Checkpoint saving mode, choices are 'linear' or 'exponential'.",
    },
    {
        "name": "ckpt_keep_limit",
        "type": int,
        "default": 5,
        "help": "Limit checkpoints to keep.",
    },
]



#registered_conf = [
#    basic_conf,
#    input_output_conf,
#    logging_conf,
#    data_preprocess_conf,
#    model_conf,
#    training_conf,
#    cyclic_learning_conf,
#    ckpt_conf,
#    extra_conf,
#]
from improvelib.utils import str2bool

additional_definitions = [
    {"name": "output_swarmfile_dir",
     "type": str,
     "default": './',
     "help": "Path to save the swarmfiles."
    },
    {"name": "model_name",
     "type": str,
     "default": 'PathDSP',
     "help": "Name of the deep learning model"
    },
    {"name": "model_scripts_dir",
     "type": str,
     "default": './', 
     "help": "Path to the model repository"
    },
    {"name": "model_environment",
     "type": str,
     "default": '',
     "help": "Name of your model conda environment"
    },
    {"name": "lca_splits_dir",
     "type": str,
     "default": './',
     "help": "Path to LCA splits"
    },
    {"name": "dataset",
     "type": str,
     "default": 'CCLE',
     "help": "Dataset to use."
    },
    {"name": "split_nums",
     "nargs" : "+",
     "type": str,
     "default": ['0', '1'],
     "help": "Split of the datasets for LCA"
    },
    {"name": "y_col_name",
     "type": str,
     "default": 'auc',
     "help": "y col name"
    },
    {"name": "epochs",
     "type": int,
     "default": None,
     "help": "Number of epochs"
    },
    {"name": "cuda_name",
     "type": str,
     "default": None,
     "help": "Cuda device name.",
    },
    {"name": "swarm_file_prefix",
     "type": str,
     "default": None,
     "help": "Prefix for swarm files. If none is specfied, they will be prefixed with <model_name>_<dataset>_.",
    },
    {"name": "input_supp_data_dir",
     "type": str,
     "default": None,
     "help": "Supp data dir, if required by the model."
    },
    ]

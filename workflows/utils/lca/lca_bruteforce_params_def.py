from improvelib.utils import str2bool

additional_definitions = [
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
    {"name": "epochs",
     "type": int,
     "default": 10,
     "help": "Number of epochs"
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
    {"name": "uses_cuda_name",
     "type": str2bool,
     "default": True,
     "help": "Change to false if the model doesn't have a cuda_name parameter."
    },
    {"name": "cuda_name",
     "type": str,
     "default": "cuda:0",
     "help": "Cuda device name.",
    },

    ]

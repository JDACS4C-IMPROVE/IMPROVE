from improvelib.utils import str2bool

additional_definitions = [
    {"name": "model_name",
     "type": str,
     "default": 'lgbm',
     "help": "Name of the deep learning model"
    },
    {"name": "model_scripts_dir",
     "type": str,
     "default": './', 
     "help": "Path to the model repository"
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
    {"name": "preprocess_args",
     "type": str,
     "default": '{}',
     "help": "Additional parameters for preprocess. Should be a dictionary, for example {'supp_input_data_dir': 'supp_data', 'y_col_name': 'loewe'}."
    },
    {"name": "train_args",
     "type": str,
     "default": '{}',
     "help": "Additional parameters for train. Should be a dictionary, for example {'epochs': 100, 'y_col_name': 'loewe'}."
    },
    {"name": "infer_args",
     "type": str,
     "default": '{}',
     "help": "Additional parameters for infer. Should be a dictionary, for example {'cuda_name': 0, 'y_col_name': 'loewe'}."
    },
    ]

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
    {"name": "source_datasets",
     "nargs" : "+",
     "type": str,
     "default": ['CCLE'],
     "help": "source_datasets for cross study analysis"
    },
    {"name": "target_datasets",
     "nargs" : "+",
     "type": str,
     "default": ["CCLE", "gCSI"],
     "help": "target_datasets for cross study analysis"
    },
    {"name": "split_nums",
     "nargs" : "+",
     "type": str,
     "default": ['0'],
     "help": "Split of the source datasets for CSA"
    },
    {"name": "only_cross_study",
     "type": str2bool,
     "default": False,
     "help": "If only cross study analysis is needed"
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
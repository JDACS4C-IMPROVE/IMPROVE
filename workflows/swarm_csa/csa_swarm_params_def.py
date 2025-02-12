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
    {"name": "model_environment",
     "type": str,
     "default": '',
     "help": "Name of your model conda environment"
    },
    {"name": "epochs",
     "type": int,
     "default": 10,
     "help": "Number of epochs"
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
    {"name": "swarm_file_prefix",
     "type": str,
     "default": None,
     "help": "Prefix for swarm files. If none is specfied, they will be prefixed with <model_name>_<dataset>_.",
    },
    ]
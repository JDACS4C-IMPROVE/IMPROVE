from improvelib.utils import str2bool

additional_definitions = [
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
    {"name": "split",
     "nargs" : "+",
     "type": str,
     "default": ['0'],
     "help": "Split of the source datasets for CSA"
    },
    {"name": "only_cross_study",
     "type": bool,
     "default": False,
     "help": "If only cross study analysis is needed"
    },
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
    {"name": "model_environment",
     "type": str,
     "default": '',
     "help": "Name of your model conda environment"
    },
    {"name": "available_accelerators",
     "nargs" : "+",
     "type": str,
     "default": ["0", "1"],
     "help": "GPU IDs to assign jobs"
    },
    {"name": "parsl_config_file",
     "type": str,
     "default": './parsl_configs/lambda.py',
     "help": "Path to Parsl configuration file."
    },
]

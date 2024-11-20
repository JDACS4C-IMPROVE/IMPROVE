
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
     "default": 'graphdrp',
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
    {"name": "hyperparameters_file",
     "type": str,
     "default": 'hyperparameters_default.json',
     "help": "json file containing optimized hyperparameters per dataset"
    },
    {"name": "epochs",
     "type": int,
     "default": 10,
     "help": "Number of epochs"
    },
    {"name": "available_accelerators",
     "nargs" : "+",
     "type": str,
     "default": ["0", "1"],
     "help": "GPU IDs to assign jobs"
    },
    {"name": "use_singularity",
     "type": bool,
     "default": True,
     "help": "Do you want to use singularity image for running the model?"
    },
    {"name": "singularity_image",
     "type": str,
     "default": '',
     "help": "Singularity image file of the model"
    },
    {"name": "input_supp_data_dir",
     "type": str,
     "default": '',
     "help": "Dir containing supplementary data in addition to benchmark data (usually model-specific data)."
    }
]

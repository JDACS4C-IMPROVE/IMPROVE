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
    {"name": "val_loss",
     "type": str,
     "default": 'mse',
     "help": "Type of loss for validation"
    },
    {"name": "max_evals",
     "type": int,
     "default": 20,
     "help": "Number of evaluations"
    },
    {"name": "interactive_session",
     "type": bool,
     "default": True,
     "help": "Are you using an interactive session?"
    },
    {"name": "hyperparameter_file",
     "type": str,
     "default": './hpo_deephyper_hyperparameters.json',
     "help": "JSON file containing hyperparameters and ranges to test."
    }
    ]
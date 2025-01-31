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
    {"name": "val_metric",
     "type": str,
     "default": 'mse',
     "help": "Type of metric for validation to improve. 'mse' and 'rmse' will be minimized, all others will be maximized."
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
    }, 
    {"name": "num_gpus_per_node",
     "type": int,
     "default": 2,
     "help": "Number of GPUs per node."
    }, 
    {"name": "CBO_surrogate_model",
     "type": str,
     "default": "ET",
     "help": "Surrogate model used by the Bayesian optimization. Can be a value in ['RF', 'GP', 'ET', 'MF', 'GBRT', 'DUMMY'] or a sklearn regressor."
    }, 
    {"name": "CBO_acq_func",
     "type": str,
     "default": "UCB",
     "help": "Acquisition function used by the Bayesian optimization. Can be a value in ['UCB', 'EI', 'PI', 'gp_hedge']. Defaults to 'UCB'."
    }, 
    {"name": "CBO_acq_optimizer",
     "type": str,
     "default": "auto",
     "help": "Method used to minimze the acquisition function. Can be a value in ['sampling', 'lbfgs', 'ga', 'mixedga']. Defaults to 'auto'."
    }, 
    {"name": "CBO_acq_optimizer_freq",
     "type": int,
     "default": 10,
     "help": "Frequency of optimization calls for the acquisition function. Defaults to 10, using optimizer every 10 surrogate model updates."
    }, 
    {"name": "CBO_kappa",
     "type": float,
     "default": 1.96,
     "help": "Manage the exploration/exploitation tradeoff for the “UCB” acquisition function. Defaults to 1.96 which corresponds to 95 percent of the confidence interval."
    }
    ]

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
     "default": None,
     "help": "Number of epochs. If None, model default will be used."
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
    }, 
    {"name": "CBO_xi",
     "type": float,
     "default": 0.001,
     "help": "Manage the exploration/exploitation tradeoff of 'EI' and 'PI' acquisition function. Defaults to 0.001."
    }, 
    {"name": "CBO_update_prior",
     "type": str2bool,
     "default": False,
     "help": "Update the prior of the surrogate model with the new evaluated points. Defaults to False. Should be set to True when all objectives and parameters are continuous."
    }, 
    {"name": "CBO_update_prior_quantile",
     "type": float,
     "default": 0.1,
     "help": "The quantile used to update the prior. Defaults to 0.1."
    }, 
    {"name": "CBO_n_jobs",
     "type": int,
     "default": 1,
     "help": "Number of parallel processes used to fit the surrogate model of the Bayesian optimization. A value of -1 will use all available cores. Not used in surrogate_model if passed as own sklearn regressor. Defaults to 1."
    }, 
    {"name": "CBO_n_initial_points",
     "type": int,
     "default": 10,
     "help": "Number of collected objectives required before fitting the surrogate-model. Defaults to 10."
    }, 
    {"name": "CBO_initial_point_generator",
     "type": str,
     "default": "random",
     "help": "Sets an initial points generator. Can be either ['random', 'sobol', 'halton', 'hammersly', 'lhs', 'grid']. Defaults to 'random'."
    }, 
    {"name": "CBO_filter_failures",
     "type": str,
     "default": "min",
     "help": "Replace objective of failed configurations by 'min' or 'mean'. If 'ignore' is passed then failed configurations will be filtered-out and not passed to the surrogate model. For multiple objectives, failure of any single objective will lead to treating that configuration as failed and each of these multiple objective will be replaced by their individual 'min' or 'mean' of past configurations. Defaults to 'min' to replace failed configurations objectives by the running min of all objectives."
    }, 
    {"name": "CBO_max_failures",
     "type": int,
     "default": 100,
     "help": "Maximum number of failed configurations allowed before observing a valid objective value when filter_failures is not equal to 'ignore'. Defaults to 100."
    }, 
    ]

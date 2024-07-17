from improvelib import utils
import candle
str2bool = candle.str2bool


model_params = [
    {"name": "use_lincs",
     "type": str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
     },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression and Mordred descriptors data.",
     },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
     },
    {"name": "md_scaler_fname",
     "type": str,
     "default": "x_data_mordred_scaler.gz",
     "help": "File name to save the Mordred scaler object.",
     },
    {"name": "n_estimators",
     "type": int,
     "default": 1000,
     "help": "Number of estimators."
     },
    {"name": "max_depth",
     "type": int,
     "default": -1,
     "help": "Max depth."
     },
    # {"name": "learning_rate",
    #  "type": float,
    #  "default": 0.1,
    #  "help": "Learning rate for the optimizer."
    #  },
    {"name": "num_leaves",
     "type": int,
     "default": 31,
     "help": "Number of leaves."
     },
]

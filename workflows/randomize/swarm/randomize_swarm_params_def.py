from improvelib.utils import str2bool

additional_definitions = [
    {"name": "model_name",
     "type": str,
     "default": 'lgbm',
     "help": "Name of the deep learning model."
    },
    {"name": "model_scripts_dir",
     "type": str,
     "default": './', 
     "help": "Path to the model repository."
    },
    {"name": "model_environment",
     "type": str,
     "default": '',
     "help": "Name of your model conda environment"
    },
    {"name": "randomized_data",
     "type": str,
     "default": "{'cell_transcriptomic_file': ['default', 'cancer_gene_expression_shuffle_full_1.tsv', 'cancer_gene_expression_shuffle_full_2.tsv'], 'drug_mordred_file': ['default', 'drug_mordred_shuffle_full_1.tsv']}", 
     "help": "."
    },
    {"name": "randomized_data_dir",
     "type": str,
     "default": './', 
     "help": "Path to the randomized data."
    },
    {"name": "datasets",
     "nargs" : "+",
     "type": str,
     "default": ['CCLE'],
     "help": "Datasets to use."
    },
    {"name": "split_nums",
     "nargs" : "+",
     "type": str,
     "default": ['0'],
     "help": "Split of the datasets to use."
    },
    {"name": "split_types",
     "nargs" : "+",
     "type": str,
     "default": ['split', 'cell', 'drug'],
     "help": "Type of splits to use"
    },
    {"name": "swarm_prefix",
     "type": str,
     "default": None,
     "help": "Swarm prefix, if an alternate one is needed."
    },
    {"name": "swarm_file_prefix",
     "type": str,
     "default": None,
     "help": "Prefix for swarm files. If none is specfied, they will be prefixed with <model_name>_<dataset>_.",
    },
    {"name": "output_swarmfile_dir",
     "type": str,
     "default": './',
     "help": "Path to save the swarmfiles."
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
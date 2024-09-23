from improvelib.utils import str2bool

csa_bruteforce_params = [
    {"name": "cuda_name",
     "type": str,
     "default": "cuda:0",
     "help": "Cuda device name.",
    },
    {"name": "csa_outdir",
     "type": str,
     "default": "./run.csa.full",
     "help": "Outdir for workflow.",
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
    {"name": "model_name",
     "type": str,
     "default": 'graphdrp', ## Change the default to LGBM??
     "help": "Name of the deep learning model"
    },
    {"name": "epochs",
     "type": int,
     "default": 10,
     "help": "Number of epochs"
    },
    {"name": "uses_cuda_name",
     "type": str2bool,
     "default": True,
     "help": "Change to false if the model doesn't have a cuda_name parameter."
    },
    
]
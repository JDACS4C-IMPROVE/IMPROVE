hyperparams = [
    {"name": "batch_size",
     "type": int,
     "min": 8,
     "max": 512,
     "default": 64,
     "log_uniform": True
    },
    {"name": "learning_rate",
     "type": float,
     "min": 1e-6,
     "max": 1e-2,
     "default": 0.001,
     "log_uniform": True
    },
]

# Examples of additional hyperparamter dictionaries
# Insure that the name is a valid parameter for the model
'''
{
"name": "dropout",
"type": float,
"min": 0,
"max": 0.5,
"default": 0,
"log_uniform": False
}

{
"name": "early_stopping",
"type": "categorical",
"choices": [True, False], 
"default": False
}
'''
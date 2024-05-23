import candle
import os
from pathlib import Path
#import model_specific_config - To import model specific parameters


##TODO Replace CANDLE initialize_parameter()

fdir = Path(__file__).resolve().parent
required = None
param_csa = [
    {"name": "raw_data_dir",
     "type": str,
     "default": "raw_data",
     "help": "Data dir name that stores the raw data, including x data, y data, and splits."
    },
    {"name": "x_data_dir",
     "type": str,
     "default": "x_data",
     "help": "Dir name that contains the files with features data (x data)."
    },
    {"name": "y_data_dir",
     "type": str,
     "default": "y_data",
     "help": "Dir name that contains the files with target data (y data)."
    },
    {"name": "splits_dir",
     "type": str,
     "default": "splits",
     "help": "Dir name that contains files that store split ids of the y data file."
    },
    {"name": "improve_input_dir",
     "type": str,
     "default": Path(f"./{os.environ['IMPROVE_DATA_DIR']}/input"),
     "help": "Input directory"
    },
    {"name": "improve_output_dir",
     "type": str,
     "default": Path(f"./{os.environ['IMPROVE_DATA_DIR']}/output"),
     "help": "Input directory"
    },
    {"name": "train_split_file", # workflow
     "default": "train_split.txt",
     "type": str,
     "required": True,
     "help": "Parameter containing the train split file",
    },
    {"name": "val_split_file", # workflow
     "default": "val_split.txt",
     "type": str,
     "required": True,
     "help": "Parameter containing the validation split file",
    },
    {"name": "test_split_file", # workflow
     "default": "test_split.txt",
     "type": str,
     "required": True,
     "help": "Parameter containing the test split file",
    },
    {"name": "source_datasets",
     "type": list,
     "default": ['CCLE'],
     "help": "source_datasets for cross study analysis"
    },
    {"name": "target_datasets",
     "type": list,
     "default": ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"],
     "help": "target_datasets for cross study analysis"
    },
    {"name": "source_data_name",
     "type": str,
     "default": 'CCLE',
     "help": "Source dataset name for preprocessing"
    },
    {"name": "split",
     "type": str,
     "default": '0',
     "help": "Split number for preprocessing"
    },
    {"name": "only_cross_study",
     "type": bool,
     "default": False,
     "help": "If only cross study analysis is needed"
    }
    ]


additional_definitions = param_csa
                            #+ model_specific_config.model_params

class ImproveBenchmark(candle.Benchmark):
    """ Benchmark for Improve Models. """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Parameters
        ----------
        required: set of required parameters for the benchmark.
        additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required.update(set(required)) # This considers global framework required arguments
        if additional_definitions is not None:
            self.additional_definitions.extend(additional_definitions) # This considers global framework definitions


def initialize_parameters(filepath, default_model):
    """ Parse execution parameters from file or command line.

    Parameters
    ----------
    default_model : string
        File containing the default parameter definition.
    additional_definitions : List
        List of additional definitions from calling script.
    required: Set
        Required arguments from calling script.

    Returns
    -------
    gParameters: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

    # Build benchmark object
    par = ImproveBenchmark(
        filepath=filepath,
        defmodel=default_model,
        framework="pytorch",
        prog="Improve csa",
        desc="Framework functionality in IMPROVE"
    )

    gParameters = candle.finalize_parameters(par)

    return gParameters

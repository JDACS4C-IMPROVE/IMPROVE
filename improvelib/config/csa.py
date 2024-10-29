import os
import sys
import logging
import configparser
import importlib
import yaml
import json
from pathlib import Path

from improvelib.utils import str2bool, cast_value
from improvelib.initializer.cli import CLI
import improvelib.config.base as base

BREAK=os.getenv("IMPROVE_DEV_DEBUG", None)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL", "INFO"))

class Config(base.Config):
    """Class to handle configuration files."""
    # ConfigError = str
    config_sections = ['DEFAULT', 'CSA']

    def __init__(self) -> None:
        super().__init__()
       
        ### Set CSA specific command line options
        # 1. Get Standard Options group
        # This is the group with title 'Standard Options'
        logger.setLevel(self.log_level)

        # Set the section to CSA
        self.section = 'CSA'

        # CSA specific attributes
        self.gpu_ids = None
        self.models = None
        self.model_params = None
        self.defaults = {}
        self.parsl_config = None # Parsl config
        self.splits = None
        self.y_col_name = "auc"
        self.args = None # placeholder for args from argparse

        # Set command line options
        sog = None
        for group in self.cli.parser._action_groups:
            if group.title == 'Standard Options':
                sog = group
                break
        
        if not sog:
            logger.error("Can't find Standard Options group")
            sys.exit(1)

        sog.add_argument(
            '-g', '--gpu_ids',
            metavar='GPU_ID',
            dest="gpu_ids",
            nargs='+', 
            type=int,
            default=[0],
            help="List of GPU IDs to use."
        )

        sog.add_argument(
            '--model_params_file', '--hyperparameters_file',
            metavar='FILE',
            dest="hyperparameters_file",
            type=str,
            default=None,
            help="This file contains the dataset specific hyperparameters for the model."
        )

        sog.add_argument(
            '-e', '--epochs',
            metavar='EPOCHS',
            dest="epochs",
            type=int,
            default=1,
            help="Number of epochs to train. DEPRIECATED: Specify in model_params_per_dataset instead."
        )

        sog.add_argument(
            '-b', '--batch_size',
            metavar='BATCH_SIZE',
            dest="batch_size",
            type=int,
            help="Batch size for training. DEPRIECATED: Specify in model_params_per_dataset instead."
        )

        sog.add_argument(
            '--learning_rate',
            metavar='LEARNING_RATE',
            dest="learning_rate",
            type=float,
            help="Learning rate for training. DEPRIECATED: Specify in model_params_per_dataset instead."
        )

        sog.add_argument(
            '-x', '--cross_study_only',
            dest="cross_study_only",
            action="store_true",
            default=False,
            help="Flag for cross-study only."
        )

        sog.add_argument(
            '--model', '--model_name',
            metavar='MODEL_NAME',
            dest="model_name",
            type=str,
            default=None,
            help="Model name to use. Better path to model directory."
        )

        sog.add_argument(
            '-m', '--model_dir',
            metavar='MODEL_DIR',
            dest="model_dir",
            type=str,
            default=None,
            help="Path to model directory."
        )

        sog.add_argument(
            '--source_dataset', '--source_datasets',
            metavar='DATASET',
            dest="source_datasets",
            nargs='+',
            type=str,
            default=None,
            help="Source dataset for the workflow."
        )

        sog.add_argument(
            '--target_dataset', '--target_datasets',
            metavar='DATASET',
            dest="target_datasets",
            nargs='+',
            type=str,
            default=None,
            help="Target dataset for the workflow."
        )

        sog.add_argument(
            '--split',
            metavar='SPLITS',
            dest="split",
            nargs='+',
            type=int,
            default=[0],
            help="Number of splits for the workflow."
        )

        sog.add_argument(
            '-y', '--y_col_name',
            metavar='Y_COL_NAME',
            dest="y_col_name",
            type=str,
            default="auc",
            help="Name of the column to use as the target variable."
        )

        sog.add_argument(
            '--parsl_config_file',
            metavar='FILE',
            dest="parsl_config_file",
            type=str,
            default=None,
            help="Runtime config for parsl as a python module. Config is in parsl_config variable."
        )

        # conda_or_singularity = sog.add_mutually_exclusive_group(required=True)


        sog.add_argument(
            '-s', '--singularity',
            dest="singularity",
            action="store_true",
            default=False,
            help="Use singularity container for execution."
        )

        sog.add_argument(
            '-c', '--container',
            dest="container",
            type=str,
            default=None,
            help="URI to container image to use for execution, e.g. file:///absolute/path/to/container.sif."
        )

        sog.add_argument(
            '--conda_env',
            dest="conda_env",
            type=str,
            default=None,
            help="Conda environment to use for execution."
        )



        if not "CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    def initialize_parameters(self,
                              section='CSA',
                              default_config=None,
                              csa_config=None,
                              parsel_config=None, 
                            #   default_model=None,
                              additional_definitions=None,
                            #   required=None,
                              ):
        """Initialize parameters from command line and config file."""

        self.section = section
        p = super().initialize_parameters(pathToModelDir=None,
                                    section=self.section,
                                    default_config=default_config,
                                    additional_definitions=additional_definitions,
                                    )

        self.gpu_ids = self.cli.args.gpu_ids
        self.parsl_config_file = self.cli.args.parsl_config_file
        self.singularuty = self.cli.args.singularity
        self.container = self.cli.args.container
        self.conda_env = self.cli.args.conda_env
        self.model_name = self.cli.args.model_name
        self.model_dir = self.cli.args.model_dir
        self.defaults['epochs'] = self.cli.args.epochs
        self.defaults['batch_size'] = self.cli.args.batch_size
        self.defaults['learning_rate'] = self.cli.args.learning_rate
        self.cross_study_only = self.cli.args.cross_study_only
        self.source_datasets = self.cli.args.source_datasets
        self.target_datasets = self.cli.args.target_datasets
        self.splits = self.cli.args.split
        self.y_col_name = self.cli.args.y_col_name
        self.epochs = self.cli.args.epochs
        


        self.args = self.cli.args

        if self.parsl_config_file:
            self._load_parsl_config()
        
        if self.cli.args.hyperparameters_file:
            self.model_params_per_dataset = self.cli.args.hyperparameters_file
            self._load_model_params()
        else:
            logger.error("Model parameters file not specified.")
            sys.exit(1)
        

    def _load_parsl_config(self):
        # if the config file is a yaml file load it as yaml
        if self.parsl_config_file.endswith('.yaml'):

            # print error not supported and exit
            logger.error("YAML format is not supported for parsl_config_file")
            sys.exit(1)

            with open(self.parsl_config_file, 'r') as f:
                return yaml.safe_load(f)
        # if the config file is a json file load it as json
        elif self.parsl_config_file.endswith('.json'):

            # print error not supported and exit
            logger.error("JSON format is not supported for parsl_config_file")
            sys.exit(1)

            with open(self.parsl_config_file, 'r') as f:
                return json.load(f)
            
        # if the config file is python file import it and assign the importet parsl_config to self.parsl_config
        elif self.parsl_config_file.endswith('.py'):
            # import the parsl_config from the file

            # add the directory of the config file to the path
            sys.path.append(os.path.dirname(self.parsl_config_file))

            # get filename without extension
            filename = os.path.basename(self.parsl_config_file).split('.')[0]
            
            # import the file

            importet_parsl_config = importlib.import_module(filename)

            # assign the parsl_config to self.parsl_config
            self.parsl_config = importet_parsl_config.parsl_config

        else:
            logger.error("Unknown file format for parsl_config_file")
            sys.exit(1)

    # Load model parameters from the model_params_per_dataset file
    def _load_model_params(self):
        if not self.model_params_per_dataset:
            logger.error("Model parameters file not specified.")
            sys.exit(1)

        if not os.path.exists(self.model_params_per_dataset):
            logger.error(f"Model parameters file {self.model_params_per_dataset} not found.")
            sys.exit(1)

        with open(self.model_params_per_dataset, 'r') as f:
            self.model_params = yaml.safe_load(f)

        if self.models is None:
            self.models = self.model_params.keys()
        else:
            # add the models to the models_params
            for model in self.models:
                self.models.append(model)


if __name__ == "__main__":
    cfg = Config()


    # create path from current directory, keep everything before improvelib
    current_dir = Path(__file__).resolve().parent
    test_dir = current_dir.parents[1] / "tests"

    params = cfg.load_cli_parameters(
        test_dir / "data/additional_command_line_parameters.yml")
    # print(params)

    
    print(
        cfg.initialize_parameters(
        "./", additional_definitions=params)
    )
    print(cfg.config.items('DEFAULT', raw=False))
    print(cfg.cli.args)
    print(cfg.params)

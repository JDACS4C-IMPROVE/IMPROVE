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

        self.section = 'CSA'

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
            '-wfc', '--workflow_config_file',
            metavar='FILE',
            dest="workflow_config_file",
            type=str,
            default=None,
            help="This file contains the workflow configuration with source and target datasets and model hyperparameters. The file is in YAML or json format."
        )

        sog.add_argument(
            '-pc', '--parsl_config_file',
            metavar='FILE',
            dest="parsl_config_file",
            type=str,
            default=None,
            help="Runtime config for parsl as a python module. Config is in parsl_config variable."
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

        if self.parsl_config_file:
            self._load_parsl_config()

        

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

import os
import sys
import logging
import configparser
import yaml
import json
from pathlib import Path

from improvelib.utils import str2bool, cast_value
from improvelib.initializer.cli import CLI
import improvelib.config.base as base

BREAK=os.getenv("IMPROVE_DEV_DEBUG", None)


class Config(base.Config):
    """Class to handle configuration files."""
    # ConfigError = str
    config_sections = ['DEFAULT', 'CSA']

    def __init__(self) -> None:
        super().__init__()
       
        ### Set CSA specific command line options
        # 1. Get Standard Options group
        # This is the group with title 'Standard Options'

        self.section = 'CSA'

        sog = None
        for group in self.cli.parser._action_groups:
            if group.title == 'Standard Options':
                sog = group
                break
        
        if not sog:
            self.logger.error("Can't find Standard Options group")
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
            '-pc', '--parsl_config_file',
            metavar='FILE',
            dest="parsl_config_file",
            type=str,
            default=None,
            help="List of GPU IDs to use."
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
                                    additional_definitions=self._options,
                                    )

        self.gpu_ids = self.cli.args.gpu_ids
        self.parsl_config_file = self.cli.args.parsl_config_file
        

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

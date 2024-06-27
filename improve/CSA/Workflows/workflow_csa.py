import os
import subprocess
import warnings
from time import time
from pathlib import Path
import pandas as pd
import logging
#from ..Config import CSA, Parsl
import sys
import Config.CSA as CSA

# Parsl imports
import parsl
from parsl import python_app, bash_app, join_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = "auc"

logger = logging.getLogger(f'Start workflow')


def build_split_fname(source_data_name, split, phase):
    """ Build split file name. If file does not exist continue """
    if split=='all':
        return f"{source_data_name}_{split}.txt"
    return f"{source_data_name}_split_{split}_{phase}.txt"

@python_app  
def preprocess(source_data_name, split, params): 
    logger.info( f'\tStart data preprocessing for {source_data_name}' 
                f' split f{split}')

    for phase in ["train", "val", "test"]:
        fname = build_split_fname(source_data_name, split, phase)
        if fname not in os.listdir(params['splits_dir']):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue
        
        input_dir = params['improve_input_dir'] / source_data_name / split

        preprocess_run = ["python",
                "Paccmann_MCA_preprocess_improve.py",
                "--improve_input_dir", input_dir,
                "--source_data_name", source_data_name,
                "--split", str(split),
                "--splits_dir", splits_dir]
        
        
        
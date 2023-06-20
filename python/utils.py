import candle
import os
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath
from math import sqrt
from scipy import stats
from typing import List, Union, Optional, Tuple
import logging


logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

# Functionality in this package should be specific to IMPROVE
# Use candle_lib functionality as often as possible and
# integrate common functionality into candle_lib
# This package should be reviewd and updated regularly with
# the candle team

def make_config_template(format: str):
    return


def load(directory: Path=None, file: Path=None, categories: List[str]=None):

    if directory :
        pass
    elif file :
        logger.info("Loading data from file: %s " , file)
        cdf = candle.data_utils.load_csv_data(file, sep="\t" , return_dataframe=True)
        return cdf
    else:
        logger.error("Missing file or directory")
    
    return None
 

if __name__ == "__main__": 
    file = "/Users/me/Development/IMPROVE/Data/raw_data/x_data/cancer_DNA_methylation.tsv"
    splits="/Users/me/Development/IMPROVE/Data/raw_data/splits/CCLE_split_0_"
    df=load(file=file)

    print(type(df))
    print(len(df))
    print(type(df[0]))
    logger.info("Done")
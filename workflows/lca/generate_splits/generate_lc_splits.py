""" This script generates data splits for learning curve analysis. """

import argparse
import logging
import os
from glob import glob
from pathlib import Path
from typing import List, Optional
from pprint import pprint

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    filename='lc_splits.log', # Log file name
    level=logging.INFO, # Log level
    format='%(asctime)s - %(levelname)s - %(message)s' # Log format
)

# Console logging handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

filepath = Path(__file__).resolve().parent


def gen_lc_splits(df: pd.DataFrame,
                  min_size: int = 1,
                  max_size: Optional[int] = None,
                  n_sizes: int = 10,
                  scale: str = 'linear',
                  random_state: int = 42) -> List[List[int]]:
    """
    Generates multiple lists of row indices for progressively larger training
    sets based on different scaling methods, adjusting for specified minimum
    and maximum sizes.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns 'auc', 'Drug', 'Cell'.
        min_size (int): Minimum size for the training set, defaults to 1.
        max_size (Optional[int]): Maximum size for the training set. If None,
            set to the length of df.
        n_sizes (int): Number of training set sizes to generate, defaults to 10.
        scale (str): The scale of size increments, options are 'linear', 'log10',
            'log2', 'log'.
        random_state (int): Random seed for reproducibility, defaults to 42.

    Returns:
        List[List[int]]: Each inner list contains row indices corresponding to
            a training set size.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(min_size, int) or (max_size is not None and not isinstance(max_size, int)) or not isinstance(n_sizes, int):
        raise TypeError("min_size, max_size (if provided), and n_sizes must all be integers")
    if min_size < 1 or (max_size is not None and max_size > len(df)) or min_size >= (max_size if max_size is not None else len(df)):
        raise ValueError("min_size must be >= 1 and < max_size (if provided), and max_size must be <= length of df")
    if n_sizes < 1:
        raise ValueError("n_sizes must be a positive integer")
    if scale.lower() not in ['linear', 'log10', 'log2', 'log']:
        raise ValueError("scale must be 'linear', 'log10', 'log2', or 'log'")

    # Set max_size to the length of the data if not provided
    if max_size is None:
        max_size = len(df)

    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Create an array of shuffled row indices
    row_indices = np.arange(len(df))
    np.random.shuffle(row_indices)

    # breakpoint()
    # Generate size vectors based on the specified scale
    if scale.lower() == 'linear':
        sizes = np.linspace(min_size, max_size, n_sizes).astype(int)
        # sizes = np.linspace(min_size, max_size, n_sizes+1)[1:]
        # sizes = np.linspace(min_size, max_size, n_sizes + 1)
    else:
        if scale.lower() == 'log2':
            # sizes = 2 ** np.array(np.arange(30))[1:]
            sizes = np.logspace(np.log2(min_size), np.log2(max_size),
                                num=n_sizes, base=2).astype(int)
        elif scale.lower() == 'log10':
            # sizes = 10 ** np.array(np.arange(8))[1:]
            sizes = np.logspace(np.log10(min_size), np.log10(max_size),
                                num=n_sizes, base=10).astype(int)
        elif scale.lower() == 'log':
            pw = np.linspace(0, n_sizes - 1, num=n_sizes) / (n_sizes - 1) # Cal log scale powers
            sizes = min_size * (max_size / min_size) ** pw # Generate log sizes

        # Filter sizes to ensure they are within the specified min_size and max_size
        sizes = np.array([int(i) for i in sizes if i >= min_size and i <= max_size])

        # # Heuristic to remove the last size if the difference is too small
        # if len(sizes) > 2 and 0.5 * sizes[-3] > (sizes[-1] - sizes[-2]):
        #     sizes = sizes[:-1]

    # Create a list of lists with indices corresponding to the training set sizes
    split_lists = [row_indices[:size].tolist() for size in sizes]
    return split_lists


def save_list_to_file(input_list: List, filename: str) -> None:
    """
    Saves list elements to a specified filename (each element on a new line).

    Args:
        data (List): A list of elements to be saved in the file. 
                     Elements can be of any type (str, int, float, etc.).
        filename (str): Name of the file where the elements will be written. 
    """
    if not isinstance(input_list, list):
        raise TypeError("input_list must be a list")

    with open(filename, 'w') as f:
        for item in input_list:
            f.write(f"{item}\n")
    return None


# Arg parser
parser = argparse.ArgumentParser(description='Generate learning curve row index lists.')

parser.add_argument('--data_file_path',
                    default=None,
                    type=str,
                    help='Full path to data.')
parser.add_argument('--splits_dir',
                    default=None,
                    type=str,
                    help='Full path to data splits.')
# -----------------------------------------------------------------------------
parser.add_argument('--lc_sizes',
                    default=10,
                    type=int,
                    help='Number of subset sizes (default: 10).')
parser.add_argument('--min_size',
                    default=128,
                    type=int,
                    help='The lower bound for the subset size (default: 128).')
parser.add_argument('--max_size',
                    default=None,
                    type=int,
                    help='The upper bound for the subset size (default: None).')
parser.add_argument('--lc_step_scale',
                    default='log',
                    type=str,
                    choices=['linear', 'log', 'log2', 'log10'],
                    help='Scale of progressive sampling of subset sizes in a \
                        learning curve (log2, log, log10, linear) (default: \
                        log).')
parser.add_argument('--lc_sizes_arr',
                    nargs='+',
                    type=int,
                    default=None,
                    help='List of the actual sizes in the learning curve plot \
                        (default: None).')
parser.add_argument('--sources',
                    nargs='+',
                    type=str,
                    default=None,
                    help="List of sources (studies) to use. If None given, it uses all sources in the splits_dir (default: None).")
parser.add_argument('--n_splits',
                    default=None,
                    type=int,
                    help='The number of splits to use. If None given, uses all splits in the splits_dir, typically 10 (default: None).')
parser.add_argument('--split_type',
                    default="split",
                    type=str,
                    help="Type of split to use. 'split' for mixed-set splits, use 'cell' or 'drug' etc for other blind splits (default: 'split').")

args = parser.parse_args()
args = vars(args)
pprint(args)

# Params
data_file_path = Path(args['data_file_path'])
splits_dir = Path(args['splits_dir'])

lc_step_scale = args['lc_step_scale']
min_size = args['min_size']
max_size = args['max_size']
lc_sizes = args['lc_sizes']
sources = args['sources']
n_splits = args['n_splits']
split_type = args['split_type']
# Load y data
logging.info(f"Loading data from {data_file_path}")
ydata = pd.read_csv(data_file_path, sep='\t')

source_paths = list(Path(splits_dir).glob(f"*_{split_type}_*.txt"))
source_files = [os.path.basename(x) for x in source_paths]
if sources is None:
    sources = list(set([x.split('_')[0] for x in source_files]))
if n_splits is None:
    n_splits = max([int(x.split('_')[2]) for x in source_files]) + 1
#sources = ['CCLE', 'CTRPv2', 'gCSI', 'GDSCv1', 'GDSCv2']
#n_splits = 10

outdir = filepath / "lc_splits"
os.makedirs(outdir, exist_ok=True)
logging.info(f"Output directory: {outdir}")


# breakpoint()
for src in sources:
    for split_id in range(n_splits):

        train_split_file_name = f'{src}_split_{split_id}_train.txt'
        tr_ids = pd.read_csv(splits_dir / train_split_file_name, header=None)[0].tolist()
        ytr = ydata.loc[tr_ids]
        split_lists = gen_lc_splits(ytr, min_size=min_size, max_size=max_size,
                                    n_sizes=lc_sizes, scale=lc_step_scale)
        for ii, ids in enumerate(split_lists):
            lc_size = len(ids)
            # print(f"{src}, split {ii}, lc size {lc_size}")
            logging.info(f"{src}, split {ii}, lc size {lc_size}")
            save_list_to_file(ids, outdir / f"{src}_split_{split_id}_sz_{lc_size}.txt")

# print("\nFinished generating data splits for learning curve analysis.")
logging.info("Finished generating data splits for learning curve analysis.")

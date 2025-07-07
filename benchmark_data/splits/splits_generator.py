import pandas as pd
import numpy as np
import random
import sys
import os
import argparse
from ast import literal_eval

'''
Natasha: remaining to do
- Blind split on 2 col (for synergy drug)
- Option for blind splits to only be blind for test, not both test and val
'''

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-i', '--input_y_data', default='./y_data.tsv', help='Path to input y data file.')
    parent_parser.add_argument('-o', '--output_dir', default='./splits', help='Output directory.')
    parent_parser.add_argument('-s', '--study_col', default='study', help="Name of the column containing the study ('study' for synergy, 'source' for DRP).")
    parent_parser.add_argument('-r', '--ratio', default=(0.8, 0.1, 0.1), help='Ratios to use for splits (train, val, test).')
    parent_parser.add_argument('-n', '--n_splits', default=10, help='How many splits to create.')
    parent_parser.add_argument('-d', '--seeds', default="[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", help='List of seeds to use for randomization.')

    parser = argparse.ArgumentParser() 
    subparsers = parser.add_subparsers()

    parser_mixed = subparsers.add_parser('mixed', parents = [parent_parser])                          
    parser_mixed.set_defaults(func=generate_mixed_splits)

    parser_blind = subparsers.add_parser('blind', parents = [parent_parser])                          
    parser_blind.add_argument('-C', '--blind_col', default='DepMapID', help="Name of the column to perform blind splits on.")
    parser_blind.add_argument('-N', '--blind_name', default='cell', help="Name of the blind split (to be included in the split file name).")
    parser_blind.set_defaults(func=generate_blind_splits)
    
    args = parser.parse_args()
    func = args.func
    args = vars(args)
    args['df'] = pd.read_csv(args['input_y_data'], sep='\t')
    args['seeds'] = literal_eval(args['seeds'])
    del args['input_y_data']
    del args['func']
    func(**args)


def _save_split(path, list):
    """Saves the split file to the specified path, one item per line.

    Args:
        path (Union[str, Path]): Path to save the split file.
        list (List): List of splits to save.
    """

    with open(path, "w") as file:
        for item in list:
            file.write(str(item) + "\n") 

def _split_checks(output_dir, ratio, seeds, n_splits):
    """Checks arguments to make sure they are valid and creates output_dir.

    Args:
        output_dir (str): Output dir to create.
        ratio (tuple): Ratios to use for splits (train, val, test).
        seeds (List): List of seeds to use for randomization.
        n_splits (int): How many splits to create.
    
    Raises:
        ValueError: If n_splits is not the same as the length of seeds.
        ValueError: If the values of ratio does not total 1.
    """
    if n_splits != len(seeds):
        raise ValueError(f"n_splits is {n_splits} and the length of the list of seeds is {len(seeds)}. These must be equal.")
    elif ratio[0] + ratio[1] + ratio[2] != 1:
        raise ValueError(f"The total of the ratio must equal 1. The given ratio was {ratio}.")
    os.makedirs(output_dir, exist_ok=True)

def generate_mixed_splits(df, study_col='study', output_dir='./', ratio=(0.8, 0.1, 0.1), seeds=list(range(10)), n_splits=10):
    """Generates mixed splits and saves them in output_dir.

    Args:
        df (pd.DataFrame): y_data to split.
        study_col (str): Name of the column that contains study names to separate on (default: 'study').
        output_dir (str): Output dir to save split (default: './').
        ratio (tuple): Ratios to use for splits (train, val, test) (default: (0.8, 0.1, 0.1)).
        seeds (List): List of seeds to use for randomization (default: list(range(10))).
        n_splits (int): How many splits to create (default: 10).
    
    """
    if not output_dir.endswith('/'):
        output_dir += '/'
    studies = df[study_col].unique()
    _split_checks(output_dir, ratio, seeds, n_splits)

    for study_name in studies:  
        study_df = df[df[study_col] == study_name]
        study_indexes = study_df['split_id'].to_list()

        # save 'all' split file
        all_path = output_dir + study_name + "_all.txt"
        _save_split(all_path, study_indexes)
        # determine the size of test and val splits (remainder is train)
        test_len = np.floor(len(study_indexes) * ratio[2]).astype(int)
        val_len = np.floor(len(study_indexes) * ratio[1]).astype(int)

        for n in range(10):
            # randomize the indexes
            random.seed(seeds[n])
            ind = random.sample(study_indexes, k=len(study_indexes))

            # split study's indexes into train/val/test
            test_split = ind[0:test_len]
            val_split = ind[test_len:(test_len+val_len)]
            train_split = ind[(test_len+val_len):]

            # save split files
            train_path = output_dir + study_name + "_split_" + str(n) + "_train.txt"
            val_path = output_dir + study_name + "_split_" + str(n) + "_val.txt"
            test_path = output_dir + study_name + "_split_" + str(n) + "_test.txt"
            _save_split(train_path, train_split)
            _save_split(val_path, val_split)
            _save_split(test_path, test_split)


def generate_blind_splits(df, blind_col, blind_name, study_col='study', output_dir='./', ratio=(0.8, 0.1, 0.1), seeds=list(range(10)), n_splits=10):
    """Generates mixed splits and saves them in output_dir.

    Args:
        df (pd.DataFrame): y_data to split.
        blind_col (str): Name of the column to blind in regards to.
        blind_name (str): Name to use when saving splits to describe what it is blind in regards to.
        study_col (str): Name of the column that contains study names to separate on (default: 'study').
        output_dir (str): Output dir to save split (default: './').
        ratio (tuple): Ratios to use for splits (train, val, test) (default: (0.8, 0.1, 0.1)).
        seeds (List): List of seeds to use for randomization (default: list(range(10))).
        n_splits (int): How many splits to create (default: 10).
    
    """
    if not output_dir.endswith('/'):
        output_dir += '/'
    studies = df[study_col].unique()
    _split_checks(output_dir, ratio, seeds, n_splits)

    for study_name in studies:
        study_df = df[df[study_col] == study_name]
        targets = study_df[blind_col].unique().tolist()

        # determine the size of test and val splits (remainder is train)
        test_len = np.floor(len(targets) * ratio[2]).astype(int)
        val_len = np.floor(len(targets) * ratio[1]).astype(int)

        for n in range(10):
            # randomize the indexes
            random.seed(seeds[n])
            targ = random.sample(targets, k=len(targets))

            # split study's targets into train/val/test
            test_split_targ = targ[0:test_len]
            val_split_targ = targ[test_len:(test_len+val_len)]
            train_split_targ = targ[(test_len+val_len):]
            # split dfs for stage targets
            test_df = study_df[study_df[blind_col].isin(test_split_targ)]
            val_df = study_df[study_df[blind_col].isin(val_split_targ)]
            train_df = study_df[study_df[blind_col].isin(train_split_targ)]
            # get indexes
            test_split = test_df['split_id'].tolist()
            val_split = val_df['split_id'].tolist()
            train_split = train_df['split_id'].tolist()

            if (len(train_df) == 0) or (len(val_df) == 0) or (len(test_df) == 0):
                print(f"The dataset {study_name} has too few {blind_col} to perform this split. Skipping.")
            else:
                # save split files
                train_path = output_dir + study_name + "_" + blind_name + "_" + str(n) + "_train.txt"
                val_path = output_dir + study_name + "_" + blind_name + "_" + str(n) + "_val.txt"
                test_path = output_dir + study_name + "_" + blind_name + "_" + str(n) + "_test.txt"
                _save_split(train_path, train_split)
                _save_split(val_path, val_split)
                _save_split(test_path, test_split)
                if (len(val_df[blind_col].unique()) < 10) or (len(val_df[blind_col].unique()) < 10) or (len(test_df[blind_col].unique()) < 10):
                    print(f"Warning: {study_name} may be too small for {blind_col} splits.")
                    print("The files have been generated, but may not be appropriate for use. See below:")
                    print(f"Stage \t Number of {blind_col} \t Number of observations")
                    print(f"Train \t {len(train_df[blind_col].unique())} \t {len(train_df)}")
                    print(f"Val \t {len(val_df[blind_col].unique())} \t {len(val_df)}")
                    print(f"Test \t {len(test_df[blind_col].unique())} \t {len(test_df)}")
                    
if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import random
import sys
import os
import argparse

'''
Natasha: remaining to do
- Fix source vs study issue, probably with argpase
- Blind split on 2 col (for synergy drug)
- Argparse options to run as is on any data
- Option for blind splits to only be blind for test, not both test and val
'''

def save_split(path, list):
   with open(path, "w") as file:
    for item in list:
        file.write(str(item) + "\n") 

def _split_checks(output_dir, ratio, seeds, n_splits):
    if n_splits != len(seeds):
        print(f"n_splits is {n_splits} and the length of the list of seeds is {len(seeds)}. These must be equal. Exiting.")
        sys.exit(1)
    elif ratio[0] + ratio[1] + ratio[2] != 1:
        print(f"The total of the ratio must equal 1. The given ratio was {ratio}. Exiting.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

def generate_mixed_splits(df, output_dir='./', ratio=(0.8, 0.1, 0.1), seeds=list(range(10)), n_splits=10):
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'index_num'})
    studies = df['study'].unique()
    _split_checks(output_dir, ratio, seeds, n_splits)

    for study_name in studies:  
        study_df = df[df['study'] == study_name]
        study_indexes = study_df['index_num'].to_list()

        # save 'all' split file
        all_path = output_dir + study_name + "_all.txt"
        save_split(all_path, study_indexes)
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
            save_split(train_path, train_split)
            save_split(val_path, val_split)
            save_split(test_path, test_split)


def generate_blind_splits(df, blind_col, blind_name, output_dir='./', ratio=(0.8, 0.1, 0.1), seeds=list(range(10)), n_splits=10):
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'index_num'})
    studies = df['study'].unique()
    _split_checks(output_dir, ratio, seeds, n_splits)

    for study_name in studies:
        print(f"Starting splits for {study_name}...") 
        study_df = df[df['study'] == study_name]
        targets = study_df[blind_col].unique().tolist()
        print("total cell lines:", len(targets))
        print("targets", targets)
        #study_indexes = study_df['index_num'].to_list()

        # determine the size of test and val splits (remainder is train)
        test_len = np.floor(len(targets) * ratio[2]).astype(int)
        val_len = np.floor(len(targets) * ratio[1]).astype(int)

        for n in range(10):
            print(f"Generating split {n}...")
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
            test_split = test_df['index_num'].tolist()
            val_split = val_df['index_num'].tolist()
            train_split = train_df['index_num'].tolist()

            if (len(train_df) == 0) or (len(val_df) == 0) or (len(test_df) == 0):
                print(f"The dataset {study_name} has too few {blind_col} to perform this split. Skipping.")
            else:
                # save split files
                train_path = output_dir + study_name + "_" + blind_name + "_" + str(n) + "_train.txt"
                val_path = output_dir + study_name + "_" + blind_name + "_" + str(n) + "_val.txt"
                test_path = output_dir + study_name + "_" + blind_name + "_" + str(n) + "_test.txt"
                save_split(train_path, train_split)
                save_split(val_path, val_split)
                save_split(test_path, test_split)
                if (len(val_df[blind_col].unique()) < 10) or (len(val_df[blind_col].unique()) < 10) or (len(test_df[blind_col].unique()) < 10):
                    print(f"Warning: {study_name} may be too small for {blind_col} splits.")
                    print("The files have been generated, but may not be appropriate for use. See below:")
                    print(f"Stage \t Number of {blind_col} \t Number of observations")
                    print(f"Train \t {len(train_df[blind_col].unique())} \t {len(train_df)}")
                    print(f"Val \t {len(val_df[blind_col].unique())} \t {len(val_df)}")
                    print(f"Test \t {len(test_df[blind_col].unique())} \t {len(test_df)}")
                    

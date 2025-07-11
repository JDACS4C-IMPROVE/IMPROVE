## randomize features

import pandas as pd
import numpy as np
import random
import argparse


def shuffle_data(input_file, output_file, strategy, seed=42):
    """Shuffles any tablular data either completely, or within column. 
    ID column must be index, with column names, and tab-separated.
    Saves the resulting dataframe to the output_path.

    Args:
        input_file (str): File name of data to shuffle (including path if not in this directory).
        output_file (str): File name to save shuffled data (including path if not in this directory).
        strategy (str): Either 'full' to completely shuffle the dataframe, or 'column' to shuffle within column.
        seed (int): Random seed (default: 42).

    Raises:
        ValueError: If strategy is invalid.    
    """
    random.seed(seed)
    # strategy (str): 
    #check input
    df = pd.read_csv(input_file, header=0, index_col=0, sep='\t')
    all_df_values = pd.Series(df.values.ravel())
    df_copy = df.copy()
    if strategy == 'column':
        for c in range(df.shape[1]):
            df_copy.iloc[:, c] = random.choices(df.iloc[:, c], k=df.shape[0]) # with replacement
    elif strategy == 'full':
        for c in range(df.shape[1]):
            df_copy.iloc[:, c] = random.choices(all_df_values, k=df.shape[0]) # with replacement
    else:
        raise ValueError(f"Strategy {strategy} is invalid. Choose 'column' or 'full'.")
    df.to_csv(output_file, sep='\t', index=True)


def random_SMILES_from_file(input_file, output_file, reference_file='./DrugSpaceX-10S.smi', reference_col_name='SMILES', length_min=1, length_max=np.inf, seed=42):
    """Replaces SMILES strings with random SMILES strings pulled from a reference file, within the range of lengths specified.
    ID column must be index, with the SMILES strings in the first column, with column names, and tab-separated.
    Saves the resulting dataframe to the output_path.

    Args:
        input_file (str): File name of data to shuffle (including path if not in this directory).
        output_file (str): File name to save shuffled data (including path if not in this directory).
        reference_file (str): File to pull random SMILES from (default: './DrugSpaceX-10S.smi').
        reference_col_name (str): Name of the column in the reference file to pull SMILES from (default: 'SMILES').
        length_min (int): Minimum SMILES length to include (default: 1).
        length_max (int): Maximum SMILES length to include (default: np.inf).
        seed (int): Random seed (default: 42).
    """
    random.seed(seed)
    #check input
    df = pd.read_csv(input_file, header=0, index_col=0, sep='\t')
    smiles_col_name = df.columns[0]
    ref_df = pd.read_csv(reference_file, sep='\t')
    ref_df = ref_df[
        (ref_df[reference_col_name].str.len() >= length_min) &
        (ref_df[reference_col_name].str.len() <= length_max)
        ]
    df[smiles_col_name] = ref_df[reference_col_name].sample(n=df.shape[0]) # may need to drop index?
    df.to_csv(output_file, sep='\t', index=True)



def main():
    """Parses command line arguments and calls appropriate functions."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--input_file', default='./random.tsv', help='File name of data to shuffle (including path if not in this directory).')
    parent_parser.add_argument('--output_file', default='./random.tsv', help='File name to save shuffled data (including path if not in this directory).')
    parent_parser.add_argument('--seed', default=42, help="Random seed.")

    parser = argparse.ArgumentParser() 
    subparsers = parser.add_subparsers()

    parser_shuffle = subparsers.add_parser('shuffle', parents = [parent_parser])   
    parser_shuffle.add_argument('--strategy', default='full', help="Either 'full' to completely shuffle the dataframe, or 'column' to shuffle within column.")                       
    parser_shuffle.set_defaults(func=shuffle_data)

    parser_SMILES = subparsers.add_parser('random_SMILES', parents = [parent_parser])                          
    parser_SMILES.add_argument('--reference_file', default='./DrugSpaceX-10S.smi', help="File to pull random SMILES from.")
    parser_SMILES.add_argument('--reference_col_name', default='SMILES', help="Name of the column in the reference file to pull SMILES from.")
    parser_SMILES.add_argument('--length_mix', default=1, help="Minimum SMILES length to include.")
    parser_SMILES.add_argument('--length_max', default=np.inf, help="Maximum SMILES length to include.")
    parser_SMILES.set_defaults(func=random_SMILES_from_file)
    
    args = parser.parse_args()
    func = args.func
    args = vars(args)
    func(**args)


if __name__ == '__main__':
    main()
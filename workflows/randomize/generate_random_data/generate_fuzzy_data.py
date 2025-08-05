## fuzzy gene expression
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
import argparse



def _get_count_df(response_df, feature_df, id_col):
    #response_df = response_df[response_df["source"] == "GDSCv1"]
    response_count = response_df[id_col].value_counts() #here
    response_count = pd.DataFrame(response_count, columns=['count'])
    print("response_count:", response_count)
    response_count_withfeature = response_count.join(feature_df, how='left').reset_index()
    return response_count_withfeature

def _randomize_GE(val, count, randomize, percent):
    # find high and low of val given the percent to randomize (default is 0.1%)
    if randomize:
        val_min = val - (val * percent)
        val_max = val + (val * percent)
    else:
        val_min = val
        val_max = val
    rng = np.random.default_rng()
    # to deal with negative numbers
    if val_min.item() > val_max.item():
        val_min, val_max = val_max, val_min
    # get random numbers with given low and high (all equal to val if randomize is False)
    rand_vals = rng.uniform(low=val_min, high=val_max, size=count)
    return pd.Series(rand_vals)

def _get_single_fuzzy(df_row, id_col, randomize, percent):
    name = df_row[id_col]
    count = df_row['count']
    names = []
    for n in range(count):
        names = names + [name + "_" + str(n)]
    cols_fuzzy = []
    cols_fuzzy = cols_fuzzy + [pd.Series(names)]
    for g in range(2, df_row.size):
        val = df_row[[g]]
        #print("g: ", g)
        #print("val: ", val)
        new_val = _randomize_GE(val, count, randomize, percent)
        cols_fuzzy = cols_fuzzy + [new_val]
        #df_fuzzy[gene_name] = new_val
    df_fuzzy = pd.concat(cols_fuzzy,axis=1)
    return df_fuzzy

def _substitue_zeros(feature_df, min_val, max_val):
    if min_val < max_val:
        # check min and max
        random_df = pd.DataFrame(np.random.uniform(min_val, max_val, size=feature_df.shape), index=feature_df.index, columns=feature_df.columns)
        # Replace 0s in df with the corresponding random numbers from random_df
        feature_nonzero = feature_df.mask(feature_df == 0, random_df)
    else: 
        print(f"Min of {min_val} is not less than max of {max_val}. Not substituting zeros.")
        feature_nonzero = feature_df
    return feature_nonzero

def _determine_substitute_zeros(feature_df, zeros):
    if zeros == 'below_min':
        value_arr = feature_df.to_numpy().flatten()
        nonzero_arr = [x for x in value_arr if x != 0]
        max_val = min(nonzero_arr)
        min_val = 0
        feature_df = _substitue_zeros(feature_df, min_val, max_val)
    elif zeros == 'bottom_10':
        value_arr = feature_df.to_numpy().flatten()
        nonzero_arr = [x for x in value_arr if x != 0]
        nonzero_arr = np.sort(nonzero_arr)
        bottom10_arr = np.round(len(nonzero_arr) * 0.1).astype(int)
        max_val = max(bottom10_arr)
        min_val = min(bottom10_arr)
        feature_df = _substitue_zeros(feature_df, min_val, max_val)
    elif isinstance(zeros, list):
        # check this list
        min_val = zeros[0]
        max_val = zeros[1]
        feature_df = _substitue_zeros(feature_df, min_val, max_val)
    else:
        print(f"Invalid zeros value of {zeros}. Not substituting zeros.")
    return feature_df
    
        

def create_fuzzy(response_df, feature_df, id_col, randomize=True, percent=0.001, zeros=None):
    feature_df = _determine_substitute_zeros(feature_df, zeros)
    count_df = _get_count_df(response_df, feature_df, id_col)
    count_df = count_df.head(10) # testing only
    print("count_df", count_df)
    all_fuzzy = []
    # loop through every row
    for r in range(count_df.shape[0]):
        df_row = count_df.iloc[r]
        this_fuzzy = _get_single_fuzzy(df_row, id_col, randomize, percent)
        all_fuzzy = all_fuzzy + [this_fuzzy]
        print("done with", r, "out of", count_df.shape[0])
    all_fuzzy_df = pd.concat(all_fuzzy, axis=0)
    all_fuzzy_df = all_fuzzy_df.set_index(0)
    all_fuzzy_df.columns = feature_df.columns
    return all_fuzzy_df
    
def post_shuffle_data(df, strategy, seed=42):
    """Shuffles any tablular data either completely, or within column. 
    ID column must be index, with column names, and tab-separated.
    Returns the resulting dataframe.

    Args:
        df (pd.DataFrame): Dataframe to shuffle.
        strategy (str): Either 'full' to completely shuffle the dataframe, or 'column' to shuffle within column.
        seed (int): Random seed (default: 42).

    Raises:
        ValueError: If strategy is invalid.    
    
    Returns:
        pd.DataFrame: Shuffled data.
    """
    random.seed(seed)
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
    return df_copy

def save_df(df, path):
    try:
        import polars as pl
        print("Saving with Polars.")
        pl_df = pl.from_pandas(df)
        pl_df.write_csv(path, separator='\t')
    except:
        print("Polars not present. Using pandas to save.")
        df.to_csv(str(path), sep='\t')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_data_file', default='./response.tsv')
    parser.add_argument('--feature_file', default='./drug_mordred.tsv')
    parser.add_argument('--output_dir', default='./fuzzy_files')
    parser.add_argument('--output_file', default='fuzzy.tsv')
    parser.add_argument('--output_file_post_shuffle', default='fuzzy_shuffle.tsv')
    parser.add_argument('--id_col_name', default='improve_chem_id')
    parser.add_argument('--zeros', default=None)
    parser.add_argument('--randomize', default=False)
    parser.add_argument('--percent', default=0.01)
    parser.add_argument('--post_shuffle', default=False)
    args = vars(parser.parse_args())
    output_dir = Path(args['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    response_df = pd.read_csv(args['y_data_file'], sep='\t')
    feature_df = pd.read_csv(args['feature_file'], sep='\t', header=[0], index_col=[0])

    fuzzy_df = create_fuzzy(response_df=response_df, feature_df=feature_df, id_col=args['id_col_name'], randomize=args['randomize'], percent=args['percent'], zeros=args['zeros'])
    save_df(fuzzy_df, output_dir / args['output_file'])
    print(f"File {args['output_file']} saved to {output_dir}")
    if args['post_shuffle']:
        fuzzy_df_shuffle = post_shuffle_data(fuzzy_df, strategy='full')
        save_df(fuzzy_df_shuffle, output_dir / args['output_file_post_shuffle'])
        print(f"File {args['output_file_post_shuffle']} saved to {output_dir}")
    print("Script complete.")

if __name__ == '__main__':
    main()
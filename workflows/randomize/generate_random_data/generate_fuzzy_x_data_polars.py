## fuzzy gene expression
import os
from pathlib import Path
import polars as pl
import numpy as np
import random
import argparse
import sys



def _get_count_df(response_df, feature_df, id_col):
    '''
    pl.DataFrame
    '''
    response_count = response_df[id_col].value_counts()
    response_count.columns = [id_col, 'count']
    print("response_count:", response_count)
    response_count_withfeature = response_count.join(feature_df, how='left', on=id_col)
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
    return pl.Series(rand_vals)

def _get_single_fuzzy(df_row, id_col, randomize, percent):
    name = df_row[0]
    count = df_row[1]
    names = []
    for n in range(count):
        names = names + [name + "_" + str(n)]
    cols_fuzzy = []
    cols_fuzzy = cols_fuzzy + [pl.Series(names)]
    for g in range(2, len(df_row)):
        val = df_row[g]
        new_val = _randomize_GE(val, count, randomize, percent)
        cols_fuzzy = cols_fuzzy + [new_val]
    df_fuzzy = pl.concat(cols_fuzzy, how='horizontal')
    return df_fuzzy

def _substitue_zeros(feature_df, min_val, max_val):
    if min_val < max_val:
        # check min and max
        num_rows = feature_df.height
        expressions = []
        col_names = feature_df.columns
        for col_name in col_names[1:]:
            rands = np.random.uniform(min_val, max_val, size=(num_rows, 1))
            expressions.append(
                pl.when(pl.col(col_name) == 0)
                .then(pl.Series(rands))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
        feature_nonzero = feature_df.with_columns(expressions)
    else: 
        print(f"Min of {min_val} is not less than max of {max_val}. Not substituting zeros.")
        feature_nonzero = feature_df
    return feature_nonzero

def _determine_substitute_zeros(feature_df, zeros):
    if zeros == 'below_min':
        value_arr = feature_df.drop(feature_df.columns[0]).to_numpy().flatten()
        nonzero_arr = [x for x in value_arr if x != 0]
        max_val = min(nonzero_arr)
        min_val = 0
        feature_df = _substitue_zeros(feature_df, min_val, max_val)
    elif zeros == 'bottom_10':
        value_arr = feature_df.drop(feature_df.columns[0]).to_numpy().flatten()
        nonzero_arr = [x for x in value_arr if x != 0]
        nonzero_arr = np.sort(nonzero_arr)
        bottom10 = np.round(len(nonzero_arr) * 0.1).astype(int)
        bottom10_arr = nonzero_arr[:bottom10]
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
    
        

def create_fuzzy(response_df, feature_df, id_col, randomize=True, percent=0.001, zeros=None, use_polars=False):
    feature_df = _determine_substitute_zeros(feature_df, zeros)
    original_cols =  feature_df.columns
    count_df = _get_count_df(response_df, feature_df, id_col)
    count_df = count_df.drop_nulls()
    count_df = count_df.head(10) # testing only
    print("count_df", count_df)
    all_fuzzy = []
    r = 0 
    # loop through every row
    for row in count_df.iter_rows():
        this_fuzzy = _get_single_fuzzy(row, id_col, randomize, percent)
        all_fuzzy = all_fuzzy + [this_fuzzy]
        print("done with", r, "out of", count_df.height)
        r = r + 1
    print("Done creating fuzzy, concat starting.")
    all_fuzzy_df = pl.concat(all_fuzzy)
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
    if strategy == 'column':
        print("Not implement in Polars yet")
    elif strategy == 'full':
        cols_list = [df.select([df.columns[0]])]
        df = df.drop([df.columns[0]])
        all_df_values = df.to_numpy().flatten().tolist()
        for c in df.columns:
            cols_list.append(pl.Series(c, random.choices(all_df_values, k=df.height)))
        df_copy = pl.DataFrame(cols_list)
    else:
        raise ValueError(f"Strategy {strategy} is invalid. Choose 'column' or 'full'.")
    return df_copy





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
    response_df = pl.read_csv(args['y_data_file'], separator='\t')
    feature_df = pl.read_csv(args['feature_file'], separator='\t')

    fuzzy_df = create_fuzzy(response_df=response_df, feature_df=feature_df, id_col=args['id_col_name'], randomize=args['randomize'], percent=float(args['percent']), zeros=args['zeros'])
    fuzzy_df.write_csv(output_dir / args['output_file'], separator='\t')
    print(f"File {args['output_file']} saved to {output_dir}")
    if args['post_shuffle'] or args['post_shuffle'] == 'True' or args['post_shuffle'] == 'true':
        print("Post-shuffling data...")
        fuzzy_df_shuffle = post_shuffle_data(fuzzy_df, strategy='full')
        fuzzy_df_shuffle.write_csv(output_dir / args['output_file_post_shuffle'], separator='\t')
        print(f"File {args['output_file_post_shuffle']} saved to {output_dir}")
    print("Script complete.")

if __name__ == '__main__':
    main()
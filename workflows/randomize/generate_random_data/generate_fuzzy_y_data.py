## fuzzy gene expression
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse



def _modify_ids(df, id_cols):
    """Creates unique IDs for each column in a list.

    Args:
        df (pd.DataFrame): DataFrame to create unique ID values from.
        id_cols (List of str): List containing the column name(s) for which unique IDs need to be made.

    Returns:
        pd.DataFrame: DataFrame with unique ID values for the columns listed in id_cols.
    """
    # Loop through each ID in the given list
    for id_col in id_cols:
        print(f"Modifying IDs for {id_col}.")
        # Create df with the number of duplicated IDs 
        count_df = df[id_col].value_counts().to_frame().reset_index() #here
        count_df.columns = [id_col, 'count']
        # Iterate through the dataframe of IDs with counts
        dfs_list = []
        for index, row in count_df.iterrows():
            # Subset the main df to just the rows with the ID in for this ID with counts
            reduced_df = df[df[id_col] == row[id_col]].copy()
            # Create unique IDs
            suff_nums = list(range(row['count']))
            suffixed = [row[id_col]+'---'+str(suff) for suff in suff_nums]
            # Replace the IDs in the subsetted df with the unique IDs
            reduced_df[id_col] = suffixed
            # Add the subsetted df to the list
            dfs_list = dfs_list + [reduced_df]
        print(f"Generating dataframe of modifying IDs for {id_col}.")
        # List of subsetted dfs to one df
        df = pd.concat(dfs_list)
    print(f"IDs {id_cols} modified. Sorting dataframe.")
    # Return the df to the original order
    df = df.sort_values(by='split_id')
    return df



def save_df(df, path):
    """Saves dataframe using polars if present in the environment (faster), otherwise using pandas.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (Union[Path, str]): Path to save DataFrame (including file name).
    """
    try:
        import polars as pl
        print("Saving with Polars.")
        pl_df = pl.from_pandas(df.reset_index())
        pl_df.write_csv(path, separator='\t')
    except:
        print("Polars not present. Using pandas to save.")
        df.to_csv(str(path), sep='\t')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_data_file', default='./response.tsv')
    parser.add_argument('--output_dir', default='./fuzzy_files')
    parser.add_argument('--output_file', default='response_fuzzy.tsv')
    parser.add_argument('--id_col_names', nargs='+', default=['improve_chem_id', 'improve_sample_id'])
    args = vars(parser.parse_args())
    output_dir = Path(args['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    response_df = pd.read_csv(args['y_data_file'], sep='\t', index_col=0)
    id_cols = args['id_col_names']
    print("id_cols", id_cols)
    fuzzy_response = _modify_ids(response_df, id_cols)
    save_df(fuzzy_response, output_dir / args['output_file'])
    print(f"File {args['output_file']} saved to {output_dir}")
    print("Script complete.")

if __name__ == '__main__':
    main()
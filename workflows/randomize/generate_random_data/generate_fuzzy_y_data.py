## fuzzy gene expression
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
import argparse

def _get_counts(df, id_col):
    response_count = df[id_col].value_counts().to_frame().reset_index() #here
    response_count.columns = [id_col, 'count']
    print("response_count:", response_count)
    response_count_withdata = response_count.join(df, how='left', on=id_col)
    return response_count_withdata



def save_df(df, path):
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
    parser.add_argument('--id_col_names', default=['improve_chem_id', 'improve_sample_id'])
    args = vars(parser.parse_args())
    output_dir = Path(args['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    response_df = pd.read_csv(args['y_data_file'], sep='\t')
    feature_df = pd.read_csv(args['feature_file'], sep='\t', header=[0], index_col=[0])

    fuzzy_df = create_fuzzy(response_df=response_df, feature_df=feature_df, id_col=args['id_col_name'], randomize=args['randomize'], percent=float(args['percent']), zeros=args['zeros'])
    save_df(fuzzy_df, output_dir / args['output_file'])
    print(f"File {args['output_file']} saved to {output_dir}")
    if args['post_shuffle'] or args['post_shuffle'] == 'True' or args['post_shuffle'] == 'true':
        print("Post-shuffling data...")
        fuzzy_df_shuffle = post_shuffle_data(fuzzy_df, strategy='full')
        save_df(fuzzy_df_shuffle, output_dir / args['output_file_post_shuffle'])
        print(f"File {args['output_file_post_shuffle']} saved to {output_dir}")
    print("Script complete.")

if __name__ == '__main__':
    main()
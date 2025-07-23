import pandas as pd
import os
import json
from pathlib import Path
import argparse
from improvelib.metrics import compute_metrics
from sklearn.metrics import mean_absolute_error


def main():
    main_parser = argparse.ArgumentParser(add_help=True)
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('-i', '--input_dir', default='./')
    common_args.add_argument('-o', '--output_dir', default='./results')
    common_args.add_argument('-y', '--y_col_name', default='auc')
    common_args.add_argument('-m', '--metric_type', default='regression')
    subparsers = main_parser.add_subparsers()

    parser_lca_scores = subparsers.add_parser('random_scores', parents=[common_args])
    parser_lca_scores.set_defaults(func=lca_scores)

    args = main_parser.parse_args()
    args.func(**vars(args))




def _file_prefix(model_name, dataset):
    # Set the file name prefix
    if model_name is not None:
        if dataset is not None:
            prefix = model_name + "_" + dataset + "_"
        else:
            prefix = model_name + "_"
    else:
        if dataset is not None:
            prefix = dataset + "_"
        else:
            prefix = ""
    return prefix


def lca_scores(input_dir, output_dir, y_col_name, metric_type, **kwargs):
    input_dir_path = Path(input_dir).resolve()  # absolute path to result dir
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    scores_fpath = output_dir / "all_scores.csv"
    missing_pred_files = []
    

    dfs = []

    datasets = sorted(list(input_dir_path.glob("*")))
    datasets = [path for path in datasets if os.path.isdir(path)]
    print(datasets)
    for dataset in datasets:
        print(f"This dataset is {dataset}")
        split_types = sorted(list(dataset.glob("*")))
        split_types = [path for path in split_types if os.path.isdir(path)]
        print(split_types)


    """
    for dir_path in dirs:
        split_num = str(dir_path.name).split("_")[1]
        shard_dirs = sorted(list((dir_path).glob(f"sz_*")))
        shard_score_dict = {}  # dict (key: split id, value: dict of scores)
        for shard_dir in shard_dirs:
            preds_file_path = shard_dir / "test_y_data_predicted.csv"
            try:
                columns_to_load = [f"{y_col_name}_true", f"{y_col_name}_pred"]
                preds = pd.read_csv(preds_file_path, sep=',',
                                    usecols=columns_to_load)

                # Compute scores
                y_true = preds[f"{y_col_name}_true"].values
                y_pred = preds[f"{y_col_name}_pred"].values
                sc = compute_metrics(y_true, y_pred, metric_type=metric_type)
                sc['mae'] = mean_absolute_error(y_true, y_pred)
                shard = int(shard_dir.name.split("sz_")[1])
                shard_score_dict[shard] = sc
                # Clean
                del preds, y_true, y_pred, sc, shard

            except FileNotFoundError:
                print(f"Error: File not found! {preds_file_path}")
                missing_pred_files.append(preds_file_path)

            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        # Convert dict to df, and aggregate dfs
        shard_score_df = pd.DataFrame(shard_score_dict)
        shard_score_df = shard_score_df.stack().reset_index()
        shard_score_df.columns = ['metric', 'shard', 'value']
        shard_score_df['split'] = split_num
        if shard_score_df.empty is False:
            dfs.append(shard_score_df)

    filename = _file_prefix(model_name, dataset) + "all_scores.csv"

    # Concat dfs and save
    if not dfs:
        print("No runtimes found.")
    else:
        scores = pd.concat(dfs, axis=0)
        #scores['model'] = model_name
        if model_name is not None:
            scores['model'] = model_name
        if dataset is not None:
            scores['dataset'] = dataset
        scores.to_csv(output_dir / filename, index=False)
        del dfs

        missing_preds_filename = _file_prefix(model_name, dataset) + "missing_pred_files.txt"
        if len(missing_pred_files) > 0:
            with open(f"{output_dir}/{missing_preds_filename}", "w") as f:
                for line in missing_pred_files:
                    line = 'infer' + str(line).split('infer')[1]
                    f.write(line + "\n")
    """

    


if __name__ == '__main__':
    main()
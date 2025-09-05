import pandas as pd
import os
import json
from pathlib import Path
import argparse
from improvelib.metrics import compute_metrics
from sklearn.metrics import mean_absolute_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='./')
    parser.add_argument('-o', '--output_dir', default='./results')
    parser.add_argument('-y', '--y_col_name', default='auc')
    parser.add_argument('-m', '--metric_type', default='regression')
    parser.add_argument('-p', '--data_file_prefix', default='')
    args = parser.parse_args()
    random_scores(**vars(args))





def random_scores(input_dir, output_dir, y_col_name, metric_type, data_file_prefix, **kwargs):
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
        dataset_name = os.path.basename(os.path.normpath(dataset))
        print(f"Processing dataset {dataset_name}...")
        split_types = sorted(list(dataset.glob("*")))
        split_types = [path for path in split_types if os.path.isdir(path)]
        print(split_types)
        for split_type in split_types:
            split_type_name = os.path.basename(os.path.normpath(split_type))
            print(f"Processing split_type {split_type_name}...")
            split_nums = sorted(list(split_type.glob("*")))
            split_nums = [path for path in split_nums if os.path.isdir(path)]
            for split_num in split_nums:
                split_num_name = os.path.basename(os.path.normpath(split_num))
                print(f"Processing split_num {split_num}...")
                rands = sorted(list(split_num.glob("*")))
                rands = [path for path in rands if os.path.isdir(path)]
                for rand in rands:
                    rand_name = os.path.basename(os.path.normpath(rand))
                    preds_file_path = rand / "test_y_data_predicted.csv"
                    try:
                        columns_to_load = [f"{y_col_name}_true", f"{y_col_name}_pred"]
                        preds = pd.read_csv(preds_file_path, sep=',', usecols=columns_to_load)
                        # Compute scores
                        y_true = preds[f"{y_col_name}_true"].values
                        y_pred = preds[f"{y_col_name}_pred"].values
                        scores = compute_metrics(y_true, y_pred, metric_type=metric_type)
                        scores_df = pd.DataFrame(scores, index=[0])
                        scores_df['dataset'] = dataset_name
                        scores_df['split_type'] = split_type_name
                        scores_df['split_num'] = split_num_name
                        scores_df['rand'] = rand_name
                        if scores_df.empty is False:
                            dfs.append(scores_df)
                        # Clean
                        del preds, y_true, y_pred, scores_df

                    except FileNotFoundError:
                        print(f"Error: File not found! {preds_file_path}")
                        missing_pred_files.append(preds_file_path)

                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")  
    
    filename = data_file_prefix + "all_scores.csv"
    # Concat dfs and save
    if not dfs:
        print("No data found.")
    else:
        scores = pd.concat(dfs, axis=0)
        scores.to_csv(output_dir / filename, index=False)
        del dfs

    missing_preds_filename = data_file_prefix + "missing_pred_files.txt"
    if len(missing_pred_files) > 0:
        with open(f"{output_dir}/{missing_preds_filename}", "w") as f:
            for line in missing_pred_files:
                f.write(str(line) + "\n")


    


if __name__ == '__main__':
    main()
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
    common_args.add_argument('-M', '--model_name', default=None)
    common_args.add_argument('-d', '--dataset', default=None)
    subparsers = main_parser.add_subparsers()


    parser_runtimes = subparsers.add_parser('runtimes', parents=[common_args])
    parser_runtimes.set_defaults(func=runtimes)

    parser_lca_scores = subparsers.add_parser('lca_scores', parents=[common_args])
    parser_lca_scores.set_defaults(func=lca_scores)

    args = main_parser.parse_args()
    args.func(**vars(args))




def runtimes(input_dir, output_dir, model_name, dataset, **kwargs):
    input_dir_path = Path(input_dir).resolve()  # absolute path to result dir
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    times = []
    time_file_name = 'runtime.json'
    stage_mapping = {
    'ml_data': 'preprocess',
    'models': 'train',
    'infer': 'infer'
    }
    for stage_dir_name, stage_name in stage_mapping.items():
        stage_dir_path = Path(input_dir_path) / stage_dir_name
        dirs = sorted(list(stage_dir_path.glob("split_*")))

        missing_files = []
        times_list = []

        for dir_path in dirs:  
            shard_dirs = sorted(list((dir_path).glob(f"sz_*")))
            for shard_dir in shard_dirs:  
                runtime_file_path = shard_dir / time_file_name
                try:
                    with open(runtime_file_path, 'r') as file:
                        rr = json.load(file)
                    rr['split'] = int(dir_path.name.split("split_")[1])
                    rr['shard'] = int(shard_dir.name.split("sz_")[1])
                    times_list.append(rr)
                except FileNotFoundError:
                    print(f"File not found! {runtime_file_path}")
                    missing_files.append(runtime_file_path)

        times_df = None
        if len(times_list) > 0:
            times_df = pd.DataFrame(times_list)
            times_df = times_df.replace(to_replace='NA', value=None)
            times_df['tot_mins'] = times_df['hours'] * 60 + times_df['minutes']
            times_df['stage'] = stage_name
            if model_name is not None:
                times_df['model'] = model_name
            if dataset is not None:
                times_df['dataset'] = dataset
            times.append(times_df)

    if len(times) > 0:
        times = pd.concat(times, axis=0)
        times.to_csv(output_dir / f"runtimes.csv", index=False)

def lca_scores(input_dir, output_dir, y_col_name, metric_type, model_name, dataset, **kwargs):
    input_dir_path = Path(input_dir).resolve()  # absolute path to result dir
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    scores_fpath = output_dir / "all_scores.csv"
    missing_pred_files = []
    
    if scores_fpath.exists(): # Check if data was already aggregated
        print("Load scores")
        scores = pd.read_csv(scores_fpath, sep=',')
    else: # Aggregate data if not found
        dfs = []
        infer_dir_path = input_dir_path / "infer"
        dirs = sorted(list(infer_dir_path.glob("split_*")));  # print(dirs)
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

        # Concat dfs and save
        scores = pd.concat(dfs, axis=0)
        #scores['model'] = model_name
        if model_name is not None:
            scores['model'] = model_name
        if dataset is not None:
            scores['dataset'] = dataset
        scores.to_csv(output_dir / "all_scores.csv", index=False)
        del dfs

        if len(missing_pred_files) > 0:
            with open(f"{output_dir}/missing_pred_files.txt", "w") as f:
                for line in missing_pred_files:
                    line = 'infer' + str(line).split('infer')[1]
                    f.write(line + "\n")

def plot_learning_curve(**kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    print("Not implemented yet")
    # Plot the heatmap
    #plt.figure(figsize=(10, 8))
    #ax = sns.heatmap()

    # Set the labels and title
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
    #ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Set the title
    #plt.title(title)

    # Save the plot
    #plt.savefig(filepath, bbox_inches='tight', dpi=150)
    #plt.close()
    


if __name__ == '__main__':
    main()
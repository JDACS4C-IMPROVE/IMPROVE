# Examples of running models


## Dependencies
Run [download_example_model.sh](download_example_model.sh) to get curated LGBM model
Set up `IMPROVE_BENCHMARK_DIR` to point to drug response prediction benchmark

## Usage

### 1. Run preprocessing
```bash
python3 preprocessing_examples.py --input_dir $IMPROVE_BENCHMARK_DIR
```

### 2. Run training
```bash
python3 train_examples.py --data_format='.parquet' --y_col_name='auc' --y_data_suffix='example'
```
### 3. Run inference
```bash
python3 infer_examples.py --data_format='.parquet' --y_col_name='auc' --y_data_suffix='example' --model_file_format='pt' --model_file_name='model' --y_data_preds_suffix='example' --loss='r2' --json_scores_suffix='results'
```

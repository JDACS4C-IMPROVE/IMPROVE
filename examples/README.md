# Examples of running models


## Dependencies
* Run [download_example_model.sh](download_example_model.sh) to get curated LGBM model.
* Set up `PYTHONPATH` to point the main IMPROVE repo dir
* Don't set up `IMPROVE_DATA_DIR`
* Set up `IMPROVE_BENCHMARK_DIR` to point to the full path of DRP benchmark dir (e.g., `raw_data`)
```bash
export IMPROVE_BENCHMARK_DIR="/nfs/lambda_stor_01/data/apartin/projects/IMPROVE/pan-models/IMPROVE/csa_data/raw_data"
```


## Usage

### 1. Run preprocessing
```bash
python3 preprocessing_examples.py --input_dir $IMPROVE_BENCHMARK_DIR
```

### 2. Run training
```bash
python3 train_examples.py --data_format='.parquet' --y_col_name='auc' --y_data_suffix='example' --learning_rate=0.0001    
```
### 3. Run inference
```bash
python3 infer_examples.py --data_format='.parquet' --y_col_name='auc' --y_data_suffix='example' --y_data_preds_suffix='example' --json_scores_suffix='results'
```

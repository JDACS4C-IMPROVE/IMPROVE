# Postprocessing of Learning Curve Analysis (LCA)

## Overview 

This script produces aggregated randomize scores when given the output directory from any of the IMPROVE LCA workflows.

## Requirements

* [IMPROVE general environment](https://jdacs4c-improve.github.io/docs/content/INSTALLATION.html)
* a directory containing LCA output

## Installation and Setup

Create the IMPROVE general environment:

```
conda create -n IMPROVE python=3.6
conda activate IMPROVE
pip install improvelib
```


## Parameter Configuration

This workflow uses command line parameters as follows:

* `--input_dir`: Path to the LCA results (default: `'./'`).
* `--output_dir`: Path to the directory where the postprocessing will be saved (default: `'./'`).
* `--y_col_name`: The y_col_name in `test_y_data_predicted.csv` (default: `'auc'`).
* `--metric_type`: Metric type to use (default: `'regression'`).
* `--data_file_prefix`: Prefix to use for saved file name (default: `None`).



## Usage

To generate aggregate scores:
```bash
python randomize_postprocess.py  <arguments>
```
This will output a table `<data_file_prefix>all_scores.csv` in the specified `output_dir`.


## Output

The processed scores will be in `output_dir`.


# To aggregate predictions



## Requirements

Same as above.

## Installation and Setup

Same as above.

## Parameter Configuration

This workflow uses command line parameters as follows:

* `--input_dir`: Path to the LCA results (default: `'./'`).
* `--output_dir`: Path to the directory where the postprocessing will be saved (default: `'./'`).
* `--y_col_name`: The y_col_name in `test_y_data_predicted.csv` (default: `'auc'`).
* `--metric_type`: Metric type to use (default: `'regression'`).
* `--data_file_prefix`: Prefix to use for saved file name (default: `None`).
* `--id_cols`: Name(s) of the id columns. (default: `["improve_sample_id", "improve_chem_id"]`).


## Usage

To generate aggregate predictions:
```bash
python randomize_predictions.py  <arguments>
```
This will output a table `<data_file_prefix>all_predictions.csv` in the specified `output_dir`.


## Output

The aggregated predictions will be in `output_dir`.
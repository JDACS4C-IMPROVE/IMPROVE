# Postprocessing of Learning Curve Analysis (LCA)

## Overview 

This script produces aggregated LCA scores, run times, and plots LCA mean absolute error curves when given the output directory from any of the IMPROVE LCA workflows (bruteforce, swarm).

## Requirements

* [IMPROVE general environment](https://jdacs4c-improve.github.io/docs/content/INSTALLATION.html)
* seaborn (for plotting only)
* matplotlib (for plotting only)
* a directory containing LCA output

## Installation and Setup

Create the IMPROVE general environment:

```
conda create -n IMPROVE python=3.6
conda activate IMPROVE
pip install improvelib
```

If you wish to use the included plotting functionality, install seaborn and matplotlib:

```
conda install seaborn matplotlib
```

## Parameter Configuration

This workflow uses command line parameters. The first (positional) parameter (`runtimes`, `lca_scores`, `plot_learning_curve`, or `whole_analysis`) specifies the analysis to run. Other optional parameters are as follows:

* `--input_dir`: Path to the LCA results (default: `'./'`).
* `--output_dir`: Path to the directory where the postprocessing will be saved (default: `'./'`).
* `--y_col_name`: The y_col_name in `test_y_data_predicted.csv` (default: `'auc'`).
* `--metric_type`: Metric type to use (default: `'regression'`).
* `--model_name`: Name of the model, if you would like it saved in the file name / data / title of the plot (default: `None`).
* `--dataset`: Name of the dataset, if you would like it saved in the file name / data / title of the plot (default: `None`).


## Usage

To generate run-time analysis:
```bash
python lca_postprocess.py runtimes <arguments>
```
This will output a table `runtimes.csv` in the specified `output_dir`.

To generate aggregate scores:
```bash
python lca_postprocess.py lca_scores <arguments>
```
This will output a table `all_scores.csv` in the specified `output_dir`.

To generate learning curve plot:
```bash
python lca_postprocess.py plot_learning_curve <arguments>
```
This will use `all_scores.csv` the specified `output_dir` and output a plot `fig.png` in the specified `output_dir`.

To run all analyses:
```bash
python lca_postprocess.py whole_analysis <arguments>
```
This will run the run-time analysis, aggregate scores, and plot the learning curve.


## Output

The processed results will be in `output_dir` as follows:

```
output_dir/
├── <model>_<dataset>_all_scores.csv
├── <model>_<dataset>_fig.png
└── <model>_<dataset>_runtimes.csv
```


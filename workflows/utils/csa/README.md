# Post-processing results from cross-study analysis (CSA)

## Overview

This README provides an overview of the post-processing pipeline designed for analyzing and evaluating Cross-Study Analysis (CSA) results. The pipeline generates metrics, visualizations, and summaries, comparing model predictions to ground truth values of test sets. This post-processing offers a comprehensive evaluation of model prediction performance within and across datasets.

## Installation

Please refer to the main [README](https://github.com/JDACS4C-IMPROVE/IMPROVE/blob/develop/README.md#Installation) for detailed installation instructions, including setting up the environment and installing required dependencies.

## Usage

CSA experiment results are often stored in a model directory (e.g., `LGBM/run.csa.small`). Note that CSA post-processing only requires the raw prediction results obtained via inference runs. You can launch post-processing pipeline as follows:

```
MODEL_DIR=LGBM
CSA_EXPERIMENT_DIR=run.csa.small
python csa_postproc.py --res_dir ${MODEL_DIR}/${CSA_EXPERIMENT_DIR} --model_name ${MODEL_DIR} --y_col_name auc
```

## Example usage

In this example, we demontrate how launch the post-processing pipeline with the example data provided in [./LGBM/run.csa.small](./LGBM/run.csa.small).

### 1. Clone IMPROVE repo
Clone the `IMPROVE` repository to a directory of your preference.

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

### 2. Set PYTHONPATH
Assuming you are currently inside IMPROVE directory, run the following. This adds IMPROVE repo to `PYTHONPATH`.

```bash
source setup_improve.sh
```

### 3. Run post-processing
Assuming the CSA results are located in `IMPROVE/workflows/utils/csa/LGBM/run.csa.small`, run the post-processing script:

```
python workflows/utils/csa/csa_postproc.py --res_dir workflows/utils/csa/${MODEL_DIR}/${CSA_EXPERIMENT_DIR} --model_name ${MODEL_DIR} --y_col_name auc
```

**Argument Definitions**
* `res_dir (required)`: Path to the directory containing the results. This should include the predicted and ground truth values. An example is provided in [./LGBM/run.csa.small](./LGBM/run.csa.small).

* `model_name (required)`: Name of the prediction model (e.g., GraphDRP, DeepCDR). This name will be used in the output summaries and visualizations. 

* `y_col_name (optional)`: Name of the column representing the target variable that the model predicts. The default is `auc` (`auc` represents the area under the dose response curve of a drug viability experiment).

* `outdir (optional)`: Directory to save the post-processing results, including metrics, summaries, and visualizations. If not specified, results will be saved in the current directory (`./`).

## Output Files

This pipeline generates in the specified output directory (`outdir`):

1. `all_scores.csv`: Contains detailed performance metrics (e.g., mse, rmse, pcc, scc, r2) for each study comparison.
  - `met`: The prediction performance metric name (e.g., r2).
  - `split`: Intigers indicating the data splits (e.g., 0, 1, etc.).
  - `value`: The calculated metric value for that split.
  - `src` and `trg`: The source and target dataset names (e.g., CCLE, GDSCv2, gCSI), indicating comparisons within or across datasets.

2. `densed_csa_table.csv`: This file provides a summary of the mean and standard deviation for each metric, categorized into `within` and `cross` analyses. The `within` summary statistic is calculated as the mean of values along the diagonal of the CSA results table, representing performance within the same dataset. The `cross` summary statistic, on the other hand, is calculated as the mean of values on the off-diagonal, capturing performance across different datasets.
  - `met`: The metric name.
  - `mean`: The mean value of the metric for within-dataset or cross-dataset.
  - `std`: The standard deviation of the metric, representing variability across studies.
  - `summary`: Either "within" (comparisons within the same dataset) or "cross" (comparisons across different datasets).
 
3.`<metric>_scores.csv`: Files containing detailed prediction performance scores for each metric for different datasets.

4.`<metric>_mean_csa_table.csv`: Files containing the mean of prediction performance scores for a specific metric across all studies.

5.`<metric>_std_csa_table.csv`: Files containing the standard deviation of prediction performance scores for a specific metric across all studies.

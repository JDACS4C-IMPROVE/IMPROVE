# Splits Generator

## Overview
`splits_generator.py` creates randomized splits from any y_data. Either mixed or blind splits can be created. The number of splits, random seeds to use, and the ratios can all be specified.

## Requirements

* Python 3.x
* Pandas
* Numpy

## Installation and Setup

You can install the required packages using `pip`:

```bash
pip install pandas numpy
```

## Parameter Configuration

This script uses command line parameters as follows:

* `mixed`
    * `--input_y_data`: Path to input y data file (default: './y_data.tsv').
    * `--output_dir`: Output directory (default: './splits').
    * `--study_col`: Name of the column containing the study ('study' for synergy, 'source' for DRP) (default: 'study').
    * `--ratio`: Ratios to use for splits (train, val, test) (default: (0.8, 0.1, 0.1)).
    * `--n_splits`: How many splits to create (default: 10).
    * `--seeds`: List of seeds to use for randomization (default: "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]").
* `blind`
    * `--input_y_data`: Path to input y data file (default: './y_data.tsv').
    * `--output_dir`: Output directory (default: './splits').
    * `--study_col`: Name of the column containing the study ('study' for synergy, 'source' for DRP) (default: 'study').
    * `--ratio`: Ratios to use for splits (train, val, test) (default: (0.8, 0.1, 0.1)).
    * `--n_splits`: How many splits to create (default: 10).
    * `--seeds`: List of seeds to use for randomization (default: "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]").
    * `--blind_col`: Name of the column to perform blind splits on (default: 'DepMapID').
    * `--blind_name`: Name of the blind split to be included in the split file name (default: 'cell')


## Usage
For mixed splits:

```
python splits_generator.py mixed --input_y_data /PATH/TO/MY/DATA/response.tsv --output_dir /PATH/TO/MY/OUTPUT/ --study_col source 
```

For blind splits:

```
python splits_generator.py blind --input_y_data /PATH/TO/MY/DATA/response.tsv --output_dir /PATH/TO/MY/OUTPUT/ --study_col source --blind_col improve_sample_id --blind_name cell
```


## Output

For each study in the y data, `n_splits` will be generated, each with a train, val, and test set. The naming convention for mixed splits is `<STUDY>_split_<SPLIT>_<STAGE>.txt`, for example `CCLE_split_0_train.txt`. For blind splits the naming convention is `<STUDY>_<BLIND_NAME_<SPLIT>_<STAGE>.txt`. The splits will be save in the specified output_dir.
# Cross-Study Analysis (CSA) with Swarm

## Overview 

The scripts contained here create swarm files that can be directly run on a system with Swarm to produce Cross Study Analysis results that are standardized and compatible with IMPROVE CSA postprocessing scripts.

## Requirements

* [IMPROVE general environment](https://jdacs4c-improve.github.io/docs/content/INSTALLATION.html)
* [Swarm](https://hpc.nih.gov/apps/swarm.html)
* An IMPROVE-compliant model and its environment

## Installation and Setup

Create the IMPROVE general environment:

```bash
conda create -n IMPROVE python=3.6
conda activate IMPROVE
pip install improvelib
```

Download the IMPROVE repo containing the workflows:

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
```

Download the repo for the model of choice:

```bash
cd <WORKING_DIR>
git clone https://github.com/JDACS4C-IMPROVE/<MODEL>
```

Download the benchmark dataset:
```bash
# Give examples here
```

Create a Conda environment path for the model in the model directory (or location of your choice):

```bash
conda env create -f <MODEL_ENV>.yml -p ./<MODEL_ENV_NAME>/
```


## Parameter Configuration

This workflow uses IMPROVE parameter handling. You should create a config file following the template of `csa_swarm_params.ini` with the parameters appropriate for your experiment. Parameters may also be specified on the command line.

* `input_dir`: Path to benchmark data. If using a DRP model with standard setup, this should be `./csa_data/raw_data`
* `output_dir`: Path to save the CSA results. Note that the swarm files are not written here, they are written to `output_swarmfile_dir`.
* `model_name`: Name of the model as used in scripts (i.e. `<model_name>_preprocess_improve.py`). Note that this is case-sensitive.
* `model_scripts_dir`: Path to the model repository as cloned above. Can be an absolute or relative path.
* `model_environment`: Name of the model environment as created above. Can be a path, or just the name of environment directory if it is located in `model_scripts_dir`.
* `source_datasets`: List of strings of the names of source datasets (default: ['CCLE']).
* `target_datasets`: List of strings of the names of target datasets (default: ["CCLE", "gCSI"]).
* `split_nums`: List of strings of the numbers of splits (default: ['0']).
* `only_cross_study`: Boolean indicating whether to omit within-study comparisions (default: False).
* `swarm_file_prefix`: Prefix for swarm files. If none is specfied, they will be prefixed with '<model_name>_<dataset>_'.
* `output_swarmfile_dir`: Path to save the swarm files (default: './').
* `preprocess_args`: Dictionary of additional model preprocess parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).
* `train_args`: Dictionary of additional model train parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).
* `infer_args`: Dictionary of additional model infer parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).



## Usage

Activate the IMPROVE environment:

```bash
conda activate IMPROVE
```

Create the swarm files with your configuration files:

```bash
python csa_swarm.py --config <YOUR_CONFIG_FILE>.ini
```

Run the swarm files (example usage for Biowulf):

```bash
swarm --merge-output -g 30 --time-per-command 00:10:00 -J model_preprocess preprocess.swarm
```

```bash
swarm --merge-output --partition=gpu --gres=gpu:k80:1 -g 60 --time-per-command 06:00:00 -J model_train train.swarm
```

```bash
swarm --merge-output --partition=gpu --gres=gpu:k80:1 -g 60 --time-per-command 00:30:00 -J model_train infer.swarm
```

You may need to change the memory (`-g`) and time (`--time-per-command`) allocations for your model. The `-J` flag labels the standard out and may be omitted. It may be useful to add job dependencies for train and infer with `--dependency afterany:<JOBID>`. See Biowulf documentation for Swarm [here](https://hpc.nih.gov/apps/swarm.html).

## Output

The output will be in the specified `output_dir` with the following structure (with the used source and target names and splits):
```
output_dir/
├── infer
│   ├── source[0]-target[0]
│   │   ├── split_0
│   │   │   ├── param_log_file.txt
│   │   │   ├── test_scores.json
│   │   │   └── test_y_data_predicted.csv
│   │   ├── split_1
│   │   ├── ...
│   │   └── split_9
│   ├── source[0]-target[1]
│   ├── ...
│   └── source[4]-target[4]
├── ml_data
│   ├── source[0]-target[0]
│   │   ├── split_0
│   │   │   ├── param_log_file.txt
│   │   │   ├── train_y_data.csv
│   │   │   ├── val_y_data.csv
│   │   │   ├── test_y_data.csv
│   │   │   └── train/val/test x_data, and other files per model
│   │   ├── split_1
│   │   ├── ...
│   │   └── split_9
│   ├── source[0]-target[1]
│   ├── ...
│   └── source[4]-target[4]
└── models
    ├── source[0]
    │   ├── split_0
    │   │   ├── param_log_file.txt
    │   │   ├── val_scores.json
    │   │   ├── val_y_data_predicted.csv
    │   │   └── trained model file
    │   ├── split_1
    │   ├── ...
    │   └── split_9
    ├── source[1]
    ├── ...
    └── source[4]
 ```

 We recommend using the postprocessing script for CSA to aggregate the results. See [here](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/develop/workflows/csa/postprocess).
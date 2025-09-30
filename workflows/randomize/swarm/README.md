# Randomization Analysis with Swarm

## Overview 

The scripts contained here create swarm files that can be directly run on a system with Swarm to produce Randomize Analysis results that are standardized and compatible with IMPROVE Randomize postprocessing scripts.

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

This workflow uses IMPROVE parameter handling. You should create a config file following the template of `lca_swarm_params.ini` with the parameters appropriate for your experiment. Parameters may also be specified on the command line.

* `input_dir`: Path to benchmark data. If using a DRP model with standard setup, this should be `./csa_data/raw_data`
* `output_dir`: Path to save the Randomize results. Note that the swarm files are not written here, they are written to `output_swarmfile_dir`.
* `model_name`: Name of the model as used in scripts (i.e. `<model_name>_preprocess_improve.py`). Note that this is case-sensitive.
* `model_scripts_dir`: Path to the model repository as cloned above. Can be an absolute or relative path.
* `model_environment`: Name of the model environment as created above. Can be a path, or just the name of environment directory if it is located in `model_scripts_dir`.
* `randomized_data`: Dictionary containing the feature files to test and the names of the randomized feature files ('default' can be used for the standard feature as determined in the model config file) (default: "{'cell_transcriptomic_file': ['default', 'cancer_gene_expression_shuffle_full_1.tsv', 'cancer_gene_expression_shuffle_full_2.tsv'], 'drug_mordred_file': ['default', 'drug_mordred_shuffle_full_1.tsv']}").
* `randomized_data_dir`: Path to the randomized data.
* `datasets`: Name of the datasets as used in the split names. Note that this is case-sensitive (default: ['CCLE']).
* `split_nums`: List of strings of the numbers of splits (default: ['0']).
* `split_types`: List of strings of split types (default: ['split', 'cell', 'drug']).
* `swarm_prefix`: Alternate string to prefix to the swarm command. This usually consists of anything needed to set up the enviroment before calling the script. If none is specified f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {model_env} ; export PYTHONPATH=../../../../IMPROVE ; " will be used (default: None).
* `swarm_file_prefix`: Prefix for swarm files. If none is specfied, they will be prefixed with <model_name>_<dataset>_ (default: None).
* `output_swarmfile_dir`: Path to save the swarm files (default: './').
* `preprocess_args`: Dictionary of additional model preprocess parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).
* `train_args`: Dictionary of additional model train parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).
* `infer_args`: Dictionary of additional model infer parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).


## Usage

Activate the IMPROVE environment:

```
conda activate IMPROVE
```

Create the swarm files with your configuration files:

```
python randomize_swarm.py --config <yourconfig.ini>
```

Run the swarm files (example usage for Biowulf):

```
swarm --merge-output -g 30 --time-per-command 00:10:00 preprocess.swarm
```

```
swarm --merge-output --partition=gpu --gres=gpu:k80:1 -g 60 --time-per-command 06:00:00 train.swarm
```

```
swarm --merge-output --partition=gpu --gres=gpu:k80:1 -g 60 --time-per-command 00:30:00 infer.swarm
```

You may need to change the memory (`-g`) and time (`--time-per-command`) allocations for your model. The `-J` flag labels the standard out and may be omitted. It may be useful to add job dependencies for train and infer with `--dependency afterany:<JOBID>`. See Biowulf documentation for Swarm [here](https://hpc.nih.gov/apps/swarm.html).


## Output

The output will be in the specified `output_dir`. We recommend using the postprocessing script for Randomize to aggregate the results. See here.













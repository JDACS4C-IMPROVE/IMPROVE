# Cross-Study Analysis (CSA) with Parsl 
 
## Overview 

The scripts contained here allow running the Cross-Study Analysis with Parsl (parallelized) and produce results that are standardized and compatible with IMPROVE CSA postprocessing scripts.


## Requirements

* [IMPROVE base environment](https://jdacs4c-improve.github.io/docs/content/INSTALLATION.html)
* [Parsl](https://parsl.readthedocs.io/en/stable/index.html)
* An IMPROVE-compliant model and its environment


## Installation and Setup

Create the IMPROVE general environment:

```bash
#conda create -n parsl parsl numpy pandas scikit-learn pyyaml -y
conda create -n parsl python=3.6 parsl
conda activate parsl
pip install improvelib
```

Clone the IMPROVE repo (you'll need the workflows contained in this repo):

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
```

Clone the model you want to use for CSA (model should be IMPROVE-comliant):

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

This workflow uses IMPROVE parameter handling. You should create a config file following the template of `csa_parsl_params.ini` with the parameters appropriate for your experiment. Parameters may also be specified on the command line.

* `input_dir`: Path to benchmark data. 
* `output_dir`: Path to save the workflow results.
* `model_name`: Name of the model as used in scripts (i.e. `<model_name>_preprocess_improve.py`). Note that this is case-sensitive.
* `model_scripts_dir`: Absolute path to the model repository as cloned above. Expects an absolute path.
* `model_environment`: Model environment as created above. Expects an absolute path.
* `source_datasets`: List of datasets to train with (default: ['CCLE']).
* `target_datasets`: List of datasets to infer on (default: ["CCLE", "gCSI"]).
* `split_nums`: List of splits to use (default: ['0']).
* `only_cross_study` Boolean indicating whether to omit within-study comparisions (default: False).
* `parsl_config_file`: Path to the Parsl config file. Configs can be found in ./parsl_configs. See [Parsl documentation](https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html) for examples of configs for other systems. (default: './parsl_configs/lambda.py')
* `available_accelerators`: GPU IDs to use (default: ["0", "1"]).
* `preprocess_args`: Dictionary of additional model preprocess parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).
* `train_args`: Dictionary of additional model train parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).
* `infer_args`: Dictionary of additional model infer parameters to include, otherwise the defaults in the model's `<MODEL>_params.ini` will be used (default: {}).


## Usage

Activate the Parsl environment:

```bash
conda activate parsl
```

Run CSA preprocessing with your configuration file:

```bash
python csa_parsl_preprocess.py --config <YOUR_CONFIG_FILE>.ini
```

Run CSA training and inference with your configuration file:

```bash
python csa_parsl_train_infer.py --config <YOUR_CONFIG_FILE>.ini
```

If submitting a job:
```bash
conda activate <MODEL_ENV>
export PYTHONPATH=/PATH/TO/YOUR/IMPROVE
python csa_parsl_preprocess.py --config <YOUR_CONFIG_FILE>.ini
python csa_parsl_train_infer.py --config <YOUR_CONFIG_FILE>.ini
```

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

### Reference
1.	Yadu Babuji, Anna Woodard, Zhuozhao Li, Daniel S. Katz, Ben Clifford, Rohan Kumar, Luksaz Lacinski, Ryan Chard, Justin M. Wozniak, Ian Foster, Michael Wilde and Kyle Chard. "Parsl: Pervasive Parallel Programming in Python." 28th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC). 2019. 10.1145/3307681.3325400

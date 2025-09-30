# Generate Randomized Data

## Overview

This repository contains the script to generate randomized data for the randomize pipeline.

## Requirements

* python 3.12.2
* joblib 1.2.0
* numpy 1.26.0
* pandas 2.1.1
* rdkit 2023.09.2
* mordred 1.2.0


## Installation and Setup
Clone the IMPROVE repo:

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
```

You can install the required packages using Conda and `benchmark_data.yml`:

```bash
conda env create -f benchmark_data.yml
```

Activate the conda environment and set the python path:

```bash
conda activate benchmark_data
export PYTHONPATH=<YOUR/PATH/TO/IMPROVE>
```

Input data:
* benchmark data for your application (here we use Drug Response Prediction).
* if creating random SMILES from file, you will need a reference file (we use `DrugSpaceX-10S.smi` which can be downloaded [here](https://drugspacex.simm.ac.cn/download/))



## Parameter Configuration

This script uses command line parameters as follows:

* `shuffle`
    * `--input_file`: File name of data to shuffle (including path if not in this directory) (default: './random.tsv').
    * `--output_file`: File name to save shuffled data (including path if not in this directory) (default: './random.tsv').
    * `--seed`: Random seed (default: 42).
    * `--strategy`: Either 'full' to completely shuffle the dataframe, or 'column' to shuffle within column (default: 'full').
* `random_SMILES`
    * `--input_file`: File name of data to shuffle (including path if not in this directory) (default: './random.tsv').
    * `--output_file`: File name to save shuffled data (including path if not in this directory) (default: './random.tsv').
    * `--seed`: Random seed (default: 42).
    * `--reference_file`: File to pull random SMILES from (default: './DrugSpaceX-10S.smi').
    * `--reference_col_name`: Name of the column in the reference file to pull SMILES from (default: 'SMILES').
    * `--length_min`: Minimum SMILES length to include (default: 1).
    * `--length_max`: Maximum SMILES length to include (default: np.inf).
    * `--also_generate_mordred`: If set to True, will also generate the mordred (default: False).
    * `--output_file_mordred`: File name to save mordred data (including path if not in this directory) (default: './mordred.tsv').


## Usage

Run `generate_random_data.py` with either `shuffle` or `random_SMILES` to generate the randomized data:

```bash
conda activate benchmark_data
export PYTHONPATH=/PATH/TO/IMPROVE/

python generate_random_data.py random_SMILES --input_file /PATH/TO/x_data/drug_SMILES.tsv --output_file /PATH/TO/OUTPUT/drug_SMILES_drugX_1.tsv --also_generate_mordred True --output_file_mordred /PATH/TO/OUTPUT/drug_mordred_drugX_1.tsv
python generate_random_data.py random_SMILES --input_file /PATH/TO/x_data/drug_SMILES.tsv --output_file /PATH/TO/OUTPUT/drug_SMILES_drugX_2.tsv --also_generate_mordred True --output_file_mordred /PATH/TO/OUTPUT/drug_mordred_drugX_2.tsv --seed 43
python generate_random_data.py random_SMILES --input_file /PATH/TO/x_data/drug_SMILES.tsv --output_file /PATH/TO/OUTPUT/drug_SMILES_drugX_3.tsv --also_generate_mordred True --output_file_mordred /PATH/TO/OUTPUT/drug_mordred_drugX_3.tsv --seed 44


python generate_random_data.py shuffle --input_file /PATH/TO/x_data/cancer_gene_expression.tsv --output_file /PATH/TO/OUTPUT/cancer_gene_expression_shuffle_full_1.tsv 
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/cancer_gene_expression.tsv --output_file /PATH/TO/OUTPUT/cancer_gene_expression_shuffle_full_2.tsv --seed 43
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/cancer_gene_expression.tsv --output_file /PATH/TO/OUTPUT/cancer_gene_expression_shuffle_full_3.tsv --seed 44

python generate_random_data.py shuffle --input_file /PATH/TO/x_data/drug_mordred.tsv --output_file /PATH/TO/OUTPUT/drug_mordred_shuffle_full_1.tsv 
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/drug_mordred.tsv --output_file /PATH/TO/OUTPUT/drug_mordred_shuffle_full_2.tsv --seed 43
python generate_random_data.py shuffle --input_file /PATH/TO/x_data/drug_mordred.tsv --output_file /PATH/TO/OUTPUT/drug_mordred_shuffle_full_3.tsv --seed 44
```


## Output

This script will output the randomized data at the path specified by `--output_file` (and `--output_file_mordred` if used).



# Generate 'fuzzy' data

## Overview

This repository contains the script to generate 'fuzzy' data, where each value is change within +/- a percentage of the original value.

## Requirements

* python 3.12.2
* joblib 1.2.0
* numpy 1.26.0
* pandas 2.1.1
* rdkit 2023.09.2
* mordred 1.2.0
* polars


## Installation and Setup
As above, but polars must be installed for generating fuzzy x data due to the size of the data generated. You can provide the percent, the strategy for dealing with zeros, and optionally also generate a shuffled dataset after fuzzying. This does one dataset at a time due to size issues when running models with this data.

Input data:
* benchmark data for your application (here we use Drug Response Prediction).
* if creating random SMILES from file, you will need a reference file (we use `DrugSpaceX-10S.smi` which can be downloaded [here](https://drugspacex.simm.ac.cn/download/))

## Parameter Configuration

This script uses command line parameters as follows:

* `generate_fuzzy_y_data.py`
    * `--y_data_file`: File name of y_data (including path if not in this directory) (default: './response.tsv').
    * `--output_dir`: Directory to save the fuzzy data (default: './fuzzy_files').
    * `--output_file`: File name to save shuffled data (default: './response_fuzzy.tsv').
    * `--id_col_names`: Name of id cols (default: '['improve_chem_id', 'improve_sample_id']').
* `random_SMILES`
    * `--y_data_file`: File name of y_data (including path if not in this directory) (default: './response.tsv').
    * `--feature_file`: File name of feature data to be fuzzied (including path if not in this directory) (default: './fuzzy_files').
    * `--output_dir`: Directory to save the fuzzy data (default: './fuzzy_files').
    * `--output_file`: File name to save fuzzy data (default: 'fuzzy.tsv').
    * `--output_file_post_shuffle`: File name to save shuffled fuzzy data (default: 'fuzzy_shuffle.tsv').
    * `--id_col_name`: Name of id col for this feature file (default: 'improve_chem_id').  


    * `--zeros`: Strategy for dealing with zeros (can be None, 'below_min', or 'bottom_10') (default: 'None').
    * `--randomize`: True to generate a randomized file as well (default: 'False').
    * `--percent`: Percent (as a decimal) to fuzzy within (default: 0.01).
    * `--dataset`: Dataset to generate (default: 'gCSI').

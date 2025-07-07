# IMPROVE
Libraries and scripts for the IMPROVE project.

## Overview
IMPROVE aims to establish methodology to systematically and rigorously compare supervised learning models.

For a detailed guide on how to use the IMPROVE library, see https://jdacs4c-improve.github.io/docs.

## Installation with pip

You can pip install improvelib with:

```bash
pip install improvelib
```

## Installation with GitHub

To install via GitHub, clone the IMPROVE repository to a directory of your preference (outside of your drug response prediction (DRP) model's directory). You can also use `git checkout` to use other versions of IMPROVE

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

You will also need to create a Conda environment with the required packages and set the PYTHONPATH:

```bash
conda create -n IMPROVE python=3.7 pandas requests tqdm typing_extensions pyyaml scikit-learn
conda activate IMPROVE
export PYTHONPATH=$PYTHONPATH:/YOUR/PATH/TO/IMPROVE
```

## Download data
Download the [cross-study analysis (CSA) benchmark data](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/) into your model's directory. For example:

```bash
./scripts/get-benchmarks $DESTINATION/csa_data/raw_data
```

The directory structure should look like this after the above steps have been completed:

```
IMPROVE
DRP_model
└── csa_data
```



## IMPROVE repository structure

The `improvelib` package follows a standard directory structure for organizing its code and resources. Here is a brief overview of the structure:

- `benchmark_data/`: This directory contains the scripts for generating and downloading benchmark data for use with `improvelib`.
- `improvelib/`: This directory contains the source code files for the `improvelib` package.
- `templates/`: This directory contains templates for curating a model with the `improvelib` package.
- `tests/`: This directory contains the unit tests for the `improvelib` package.
- `workflows/`: This directory contains the scripts for workflows with models curated with `improvelib`.
- `LICENSE`: This file contains the license information for the `improvelib` package.
- `README.md`: This file provides an overview and instructions for using the `improvelib` package.



## Examples
Two repositories demonstrating the use of the `IMPROVE library` for drug response prediction:
* https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop -- GraphDRP (deep learning model based on graph neural network)
* https://github.com/JDACS4C-IMPROVE/LGBM/tree/develop -- LightGBM model

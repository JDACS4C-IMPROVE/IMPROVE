# IMPROVE
Libraries and scripts for the IMPROVE project.

## Purpose

## Installation
Clone the `IMPROVE` repository to a directory of your preference (outside of your drug response prediction (DRP) model's directory).

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
git checkout develop
```

## Download data
Download the [cross-study analysis (CSA) benchmark data](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/) into your model's directory.
```bash
wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/
```

The directory structure should look like this after the above steps have been completed:

```
IMPROVE
DRP_model
└── csa_data
```

## Set environment variables

Specify the full path to the IMPROVE library with $PYTHONPATH and the path to the CSA data with $IMPROVE_DATA_DIR.
```bash
cd DRP_model
export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE
export IMPROVE_DATA_DIR="./csa_data/"
```

## Tutorial
For a detailed guide on how to use the IMPROVE library using an example model, LightGBM, see https://jdacs4c-improve.github.io/docs/content/unified_interface.html.

## Examples
Two repositories demonstrating the use of the IMPROVE library for drug response prediction:
* https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop -- GraphDRP (deep learning model based on graph neural network)
* https://github.com/JDACS4C-IMPROVE/LGBM/tree/develop -- LightGBM model





3. specify $PYTHOPATH and $IMPROVE_DATA_DIR environment variables.
```bash
export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE
export IMPROVE_DATA_DIR="./csa_data/"
```

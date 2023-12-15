# IMPROVE
Libraries and scripts for basic IMPROVE functionalities 

## Examples
Two repositories demonstrating the use for IMPROVE lib for drug response prediction.
* https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop -- GraphDRP (deep learning model based on graph neural network)
* https://github.com/JDACS4C-IMPROVE/LGBM/tree/master -- LightGBM model

### GraphDRP
This example shows how to the use the improve library with the GraphDRP model.

Clone the model into some directory. E.g., `/lambda_stor/data/apartin/projects/IMPROVE/pan-models`
```bash
cd /lambda_stor/data/apartin/projects/IMPROVE/pan-models
git clone https://github.com/JDACS4C-IMPROVE/GraphDRP
cd GraphDRP
git checkout develop
```

### Download data
Download the cross-study analysis (CSA) benchmark data into the model directory from https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/
```bash
wget --cut-dirs=7 -P ./ -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
```

### Environment
1. clone the `improve` repo to a directory of your preference. For example:
```bash
cd ..
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
git checkout develop
```
As a result of the commands above, the IMPROVE repo will be loacted in `/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE`

2. cd to the model repo. 
```bash
cd /lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP
```

3. specify $PYTHOPATH and $IMPROVE_DATA_DIR environment variables.
```bash
export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE
export IMPROVE_DATA_DIR="./csa_data/"
```

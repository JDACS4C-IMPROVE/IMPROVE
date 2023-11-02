# IMPROVE
Libraries and scripts for basic IMPROVE functionalities 

## Environment
1. clone this repo to a directory of your preference. For example:
```bash
cd /lambda_stor/data/apartin/projects/IMPROVE/pan-models
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
git checkout develop
```
As a result of the commands above, the IMPROVE repo will be loacted in `/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE`

2. cd to you model repo. For example:
```bash
cd /lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP
```

3. specify $PYTHOPATH and $IMPROVE_DATA_DIR environment variables.
```bash
export PYTHONPATH=$PYTHONPATH:/lambda_stor/data/apartin/projects/IMPROVE/pan-models/IMPROVE
export IMPROVE_DATA_DIR="./csa_data/"
```

## Packages

### Utils

- name: utils.py

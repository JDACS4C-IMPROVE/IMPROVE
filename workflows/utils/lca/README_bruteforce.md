## 1. Install conda environment for the curated model 
Install model, IMPROVE, and datasets:
```
cd <WORKING_DIR>
git clone https://github.com/JDACS4C-IMPROVE/<MODEL>
cd <MODEL>
source setup_improve.sh
```

Install model environment (get the name of the yml file from model repo readme):
The workflow will need to know the ./<MODEL_ENV_NAME>/.
```
conda env create -f <MODEL_ENV>.yml -p ./<MODEL_ENV_NAME>/
```

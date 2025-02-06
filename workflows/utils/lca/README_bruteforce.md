## 1. Install conda environment for the curated model 
Install model, set up environment, install IMPROVE, and download datasets:
```
cd <WORKING_DIR>
git clone https://github.com/JDACS4C-IMPROVE/<MODEL>
cd <MODEL>
conda env create -f <MODEL_ENV>.yml -n <NAME_OF_ENV>
conda activate <NAME_OF_ENV>
source setup_improve.sh
```

Set up parameters for brute force Learning Curve Analysis
```
cd <YOUR/PATH/TO/>IMPROVE/workflows/utils/lca
```

Parameters




Run brute force Learning Curve Analysis
```
python lca_bruteforce.py
```

To specify a different config file:
```
python lca_bruteforce.py --config <YOUR_CONFIG>
```


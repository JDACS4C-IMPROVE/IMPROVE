# Post-processing results from cross-study analysis


CSA run results are often stored within the model directory (e.g., GraphDRP/run.csa.full).
This example uses results from a small LGBM model run. Note that CSA post-processing only requires the inference results.


### 1. Clone IMPROVE repo
Clone the `IMPROVE library` repository to a directory of your preference.

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

### 2. Set PYTHONPATH
Here we assume you are currently inside the IMPROVE directory.

```
IMPROVE_LIB_PATH=$PWD
export PYTHONPATH=$PYTHONPATH:$IMPROVE_LIB_PATH
```

### 3. Determine the results path and run post-processing
Here we assume the CSA results are located at IMPROVE/post_process/csa/LGBM/run.csa.small

```
python post_process/csa/csa_postproc.py --res_dir post_process/csa/LGBM/run.csa.small --model_name LGBM --y_col_name auc
```

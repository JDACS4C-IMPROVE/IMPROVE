# Post-processing results from cross-study analysis

CSA experiment results are often stored within the model directory (e.g., GraphDRP/csa.results).
In this example we use results from a small LGBM model run. Note that CSA post-processing only requires the raw prediction results obtained via inference runs.


### 1. Clone IMPROVE repo
Clone the `IMPROVE library` repository to a directory of your preference.

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

### 2. Set PYTHONPATH
Here we assume you are currently inside the IMPROVE directory. This will set up `PYTHONPATH` (adds IMPROVE repo).

```bash
source setup_improve.sh
```

### 3. Determine the results path and run post-processing
Here we assume the CSA results are located at `IMPROVE/workflows/utils/csa/LGBM/run.csa.small`

```
python workflows/utils/csa/csa_postproc.py --res_dir workflows/utils/csa/LGBM/run.csa.small --model_name LGBM --y_col_name auc
```

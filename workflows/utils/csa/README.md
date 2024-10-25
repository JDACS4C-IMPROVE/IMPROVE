# Post-processing results from cross-study analysis

Results from a CSA experiment are often stored in a model directory (e.g., `GraphDRP/csa.results`).
In this example we use results from a small run of a LGBM model. Note that CSA post-processing only requires the raw prediction results obtained via inference runs.


### 1. Clone IMPROVE repo
Clone the `IMPROVE` repository to a directory of your preference.

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

### 2. Set PYTHONPATH
Assuming you are currently inside IMPROVE directory, run the following. This will set up `PYTHONPATH` (adds IMPROVE repo).

```bash
source setup_improve.sh
```

### 3. Determine the results path and run post-processing
Assuming the CSA results are located in `IMPROVE/workflows/utils/csa/LGBM/run.csa.small`, run the post-processing script:

```
python workflows/utils/csa/csa_postproc.py --res_dir workflows/utils/csa/LGBM/run.csa.small --model_name LGBM --y_col_name auc
```

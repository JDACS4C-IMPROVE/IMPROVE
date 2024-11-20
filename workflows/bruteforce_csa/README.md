# Step-by-step instructions to run cross study analysis using the brute force method

### 1. Clone the model repository
```bash
git clone <MODEL_REPO>
cd <MODEL_REPO>
git checkout <BRANCH>
```

**Requirements**:
1. Model scripts must be organized as:
    - <MODEL_NAME>_preprocess_improve.py
    - <MODEL_NAME>_train_improve.py
    - <MODEL_NAME>_infer_improve.py
2. Make sure to follow the IMPROVE lib [documentation](https://jdacs4c-improve.github.io/docs) to ensure the model is compliant with the IMPROVE framework
3. If the model uses supplemental data (i.e. author data), use the provided script in the repo to download this data (e.g. PathDSP/download_author_data.sh).

### 2. Set up model environment
Follow the steps in the model repo to set up the environment for the model and activate the model.

```bash
conda activate <MODEL_ENV>
```

### 3. Clone IMPROVE repo and set PYTHONPATH
Clone the [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/develop) repository to a directory of your preference (outside your model directory).

```bash
cd ..
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
source setup_improve.sh
```

### 4. Download benchmark data for cross study analysis

Download benchmark data to the data destination directory using [this](https://github.com/JDACS4C-IMPROVE/IMPROVE/blob/develop/scripts/get-benchmarks). For example:

```bash
./scripts/get-benchmarks ./workflows/bruteforce_csa
```

### 4. Configure the parameters for cross study analysis

#### These should be changed in csa_bruteforce_params.ini:

`model_scripts_dir` set to the path to the model directory containing the model scripts (from step 1).

`model_name` set to your model name (this should have the same capitalization pattern as your model scripts, e.g. deepttc for deepttc_preprocess_improve.py, etc).

`epochs` set to max epochs appropriate for your model, or a low number for testing.

`uses_cuda_name` set to True if your model uses cuda_name as parameter, leave as False if it does not. Also set `cuda_name` if your model uses this.

`input_supp_data_dir` add this if your model uses supplemental data. Set to the path to this folder, or the name of the folder if it is located in `model_scripts_dir`.

#### These you may want to change in csa_bruteforce_params.ini:

`csa_outdir` is `./bruteforce_output` but you can change to whatever directory you like.

`source_datasets`, `target_datasets`, and `split_nums` can be modified for testing purposes or quicker runs.

### 5. Run brute force workflow
To run with provided config file:
```
python csa_bruteforce_wf.py
```

To run with an alternate config file:
```
python csa_bruteforce_wf.py --config <YOUR_CONFIG_FILE>
```

If submitting a job:
```
conda activate <MODEL_ENV>
export PYTHONPATH=/YOUR/PATH/TO/IMPROVE
python csa_bruteforce_wf.py --config <YOUR_CONFIG_FILE>
```

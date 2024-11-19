# Step-by-step instructions to run cross study analysis using the brute force method

### 2. Clone the model repository
```bash
git clone <MODEL_REPO>
cd MODEL_NAME
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

The downloaded benchmark data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```


### These should be changed in csa_bruteforce_params.ini:

`model_scripts_dir` set to the path to the model directory containing the model scripts (from step 2).

`model_name` set to your model name (this should have the same capitalization pattern as your model scripts, e.g. deepttc for deepttc_preprocess_improve.py, etc).

`epochs` set to max epochs appropriate for your model, or a low number for testing.

`uses_cuda_name` set to True if your model uses cuda_name as parameter, leave as False if it does not. Also set `cuda_name` if your model uses this.

`input_supp_data_dir` add this if your model uses supplemental data. Set to the path to this folder, or the name of the folder if it is located in `model_scripts_dir`.

### These you may want to change in csa_bruteforce_params.ini:

`csa_outdir` is `./bruteforce_output` but you can change to whatever directory you like.

`source_datasets`, `target_datasets`, and `split_nums` can be modified for testing purposes or quicker runs.


## Running workflow

```

2. Set up conda -- Follow instructions in your repo.

3. Activate conda -- Follow instructions in your repo.

4. Set up improve
```
source setup_improve.py
```
5. Run workflow
```
python csa_bruteforce_wf.py
```

Note: If submitting a job, steps 3-5 should be in the shell script.

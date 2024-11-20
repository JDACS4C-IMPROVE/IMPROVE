
# Step-by-step instructions to run cross study analysis using Parsl on Lambda

### 1. Create and activate a conda environment to support improvelib and Parsl
```bash
conda create -n parsl parsl numpy pandas scikit-learn pyyaml -y
conda activate parsl
```

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
./scripts/get-benchmarks ./workflows/parsl_csa
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


### 5. To run cross study analysis using Parsl:
**Configuration file**:
**csa_params.ini** contains parameters necessary for the workflow (see [example_params_files](./example_params_files)). The user can change the parameters inside this configuration file.

 - `input_dir` : Path to the benchmark `raw_data` for cross study analysis. 
 - `input_supp_data_dir` : Dir containing supplementary data in addition to csa benchmark data (usually model-specific data). A common practice is to provide these data inside a dedicated dir inside model dir (e.g., PathDSP/author_data/...).
 - `output_dir` : Path to the output directory. The subdirectories in the `output_dir` will be organized as:
    - `ml_data`: Contains pre-processed data.
    - `models`: Contains trained models.
    - `infer`: Contains inference retults.
 - `source_datasets`: List of source datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2.
 - `target_datasets`: List of source datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2.
 - `split`: Splits of the source datasets for cross study analysis.
 - `hyperparameters_file`: Path to the json file containing hyperparameters per dataset. In this template two hyperparameter files are given:
    - `hyperparameters_hpo.json`: Contains hyperparameters optimized separately on all source datasets.
    - `hyperparameters_default.json`: Contains default values of the hyperparameters for the model.
 - `model_name`: Name of the model for cross study analysis.
 - `model_scripts_dir`: Path to the model directory containing the model scripts.
 - `model_environment`: Name of your model conda environment.
 - `epochs`: Number of epochs for the model.
 - `available_accelerators`: List of GPU ids to launch the jobs. The required format is: ["id1","id2"]. For example, if you want to choose GPUs 0 and 1 set available_accelerators = ["0","1"]
 - `y_col_name`: Response variable used in the model. eg: `auc`
 - `use_singularity`: True, if the model files are available in a singularity container.
 - `singularity_image`: Path to the singularity container image file (.sif) of the model scripts (optional).
 - `only_cross_study`: True, if only cross study analysis is needed without within study inferences.

**hyperparameters.json** contains a dictionary of optimized hyperparameters for the models. The key to the dictionary is the model name, which contains another dictionary with source dataset names as keys. The two hyperparameters considered for this analysis are: `batch_size` and `learning_rate`. 
The hyperparameters can be optimized using [Supervisor](https://github.com/JDACS4C-IMPROVE/HPO).

#### Execution without singularity container:
  
Make sure to change the `model_name` parameter in `csa_params.ini` to your <MODEL_NAME>.  
Change the `model_scripts_dir` parameter to the path to your model directory.   
Change the `model_environment` parameter to the name of your model conda environment.  
Make changes to `csa_params.ini` as needed for your experimenet.

Preprocesssing:
```
python workflow_preprocess.py
```

To run cross study analysis with default configuration file (csa_params.ini):
```
python workflow_csa.py
```

To run cross study analysis with a different configuration file:
```
python workflow_csa.py --config_file <CONFIG_FILE>
```

#### Execution with singularity container:
In `csa_params.ini`:
- Set use_singularity = True
- singularity_image = <PATH_TO_YOUR_SINGULARITY_CONTAINER>
- Change other parameters if needed

Preprocess the raw data:
```
python workflow_preprocess.py
```
To run cross study analysis with default configuration file (csa_params.ini):  
```
python workflow_csa.py
```
To run cross study analysis with a different configuration file:
```
python workflow_csa.py --config_file <CONFIG_FILE>
```

### Reference
1.	Yadu Babuji, Anna Woodard, Zhuozhao Li, Daniel S. Katz, Ben Clifford, Rohan Kumar, Luksaz Lacinski, Ryan Chard, Justin M. Wozniak, Ian Foster, Michael Wilde and Kyle Chard. "Parsl: Pervasive Parallel Programming in Python." 28th ACM International Symposium on High-Performance Parallel and Distributed Computing (HPDC). 2019. 10.1145/3307681.3325400

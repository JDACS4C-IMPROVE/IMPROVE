
# Step-by-step instructions to run cross study analysis using Parsl on Lambda

### 1. Clone the model repo
```
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

### 2. Clone IMPROVE repo
Clone the `IMPROVE library` repository to a directory of your preference (outside of your drug response prediction (DRP) model's directory).

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

### 3. Install Dependencies
Create and activate the conda environment (Not the model environment) to support Parsl and all other dependencies for IMPROVE
```
conda env create -f parsl_env.yml
```

If you face an error during execution you may have to do this for Parsl:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

### 4. Set PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:$<PATH_TO_IMPROVE>
```

### 5. Download cross study analysis data
Download the benchmark cross study analysis data within the model directory:
```
source download_csa.sh
```
The required benchmark data tree is shown below:
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

### 6. To run cross study analysis using Parsl:
**Configuration file**:
**csa_params.ini** contains parameters necessary for the workflow. The user can change the parameters inside this configuration file.

 - input_dir : Location of raw data for cross study analysis. 
 - output_dir : Location of the output. The subdirectories in the output_dir are organized as:
    - ml_data: Contains pre-processed data.
    - models: Contains trained models.
    - infer: Contains inference retults
 - source_datasets : List of source_datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2
 - target_datasets : List of source_datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2
 - split: Splits of the source datasets for cross study analysis.
 - hyperparameters_file: Name of the json file containing hyperparameters per dataset. In this template two hyperparameter files are given:
    - hyperparameters_hpo.json : Contains hyperparameters optimized separately on all source datasets.
    - hyperparameters_default.json : Contains default values of the hyperparameters for the model.
 - model_name: Name of the model for cross study analysis
 - model_environment: Name of your model conda environment
 - epochs: Number of epochs for the model
 - available_accelerators: List of GPU ids to launch the jobs. The required format is: ["id1","id2"]. For example, if you want to choose GPUs 0 and 1 set available_accelerators = ["0","1"]
 - y_col_name: Response variable used in the model. eg: auc
 - use_singularity: True, if the model files are available in a singularity container
 - singularity_image: Singularity image file (.sif) of the model scripts (optional)
 - only_cross_study: True, if only cross study analysis is needed without within study inferences

**hyperparameters.json** contains a dictionary of optimized hyperparameters for the models. The key to the dictionary is the model name, which contains another dictionary with source dataset names as keys. The two hyperparameters considered for this analysis are: batch_size and learning_rate. 
The hyperparameters are optimized using [Supervisor](https://github.com/JDACS4C-IMPROVE/HPO).

#### Without singularity container:

Copy all the files in this directory to your model directory.  
Make sure to change the 'model_name' parameter in csa_params.ini to your <MODEL_NAME>.  
Change the 'model_environment' variable to the name of your model conda environment.

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

#### With singularity container:
In csa_params.ini:  
    - Set use_singularity = True  
    - singularity_image = <NAME_OF_YOUR_SINGULARITY_CONTAINER>  
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
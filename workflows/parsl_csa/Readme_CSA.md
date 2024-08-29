
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
2. Make sure to follow the IMPROVE lib documentation to ensure the model is compliant with the IMPROVE framework

### 2. Clone IMPROVE repo
Clone the `IMPROVE library` repository to a directory of your preference (outside of your drug response prediction (DRP) model's directory).

```bash
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
```

### 3. Install Dependencies
Activate the model conda environment

Install Parsl (2023.6.19):
```
pip install parsl 
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
Download the cross study analysis data within the model directory:
```
source download_csa.sh
```

### 6. To run cross study analysus using PARSL on Lambda machine:

Copy all the files in this directory to your model directory. Make sure to change the 'model_name' parameter in csa_params.ini to your <MODEL_NAME>.

**csa_params.ini** contains parameters necessary for the workflow. The user can change the parameters inside this configuration file.

 - input_dir : Location of raw data for cross study analysis. 
 - output_dir : Location of the inference results
 - source_datasets : List of source_datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2
 - target_datasets : List of source_datasets for cross study analysis. With the current benchmark datasets this can be a subset of CCLE, gCSI, GDSCv1, GDSCv2 and CTRPv2
 - split: Splits of the source datasets for cross study analysis.
 - hyperparameters_file: json file containing optimized hyperparameters per dataset
 - model_name: Name of the model for cross study analysis
 - epochs: Number of epochs for the model
 - y_col_name: Response variable used in the model. eg: auc
 - use_singularity: True, if the model files are available in a singularity container
 - only_cross_study: True, if only cross study analysis is needed without within study inferences

**hyperparameters.json** contains a dictionary of optimized hyperparameters for the models. The key to the dictionary is the model name, which contains another dictionary with source dataset names as keys. The two hyperparameters considered for this analysis are: batch_size and learning_rate. 
The hyperparameters are optimized using [Supervisor](https://github.com/JDACS4C-IMPROVE/HPO).

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
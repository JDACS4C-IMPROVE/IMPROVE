import parsl
from parsl import python_app
import subprocess
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from time import time
from typing import Sequence, Tuple, Union
from pathlib import Path
import logging
import sys
import json

import csa_params_def as CSA
import improvelib.utils as frm
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig


##### CONFIG FOR LAMBDA ######
#available_accelerators: Union[int, Sequence[str]] = 8
worker_port_range: Tuple[int, int] = (10000, 20000)
retries: int = 1

config_lambda = Config(
    retries=retries,
    executors=[
        HighThroughputExecutor(
            address='127.0.0.1',
            label="htex",
            cpu_affinity="block",
            #max_workers_per_node=2, ## IS NOT SUPPORTED IN  Parsl version: 2023.06.19. CHECK HOW TO USE THIS???
            worker_debug=True,
            available_accelerators=8,  ## CHANGE THIS AS REQUIRED BY THE MACHINE
            worker_port_range=worker_port_range,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy='simple',
)

parsl.clear()
parsl.load(config_lambda)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
logger = logging.getLogger(f'Start workflow')

##############################################################################
################################ PARSL APPS ##################################
##############################################################################

@python_app  
def preprocess(inputs=[]): # 
    import warnings
    import subprocess
    import improvelib.utils as frm
    def build_split_fname(source_data_name, split, phase):
        """ Build split file name. If file does not exist continue """
        if split=='all':
            return f"{source_data_name}_{split}.txt"
        return f"{source_data_name}_split_{split}_{phase}.txt"
    params=inputs[0]
    source_data_name=inputs[1]
    split=inputs[2]

    split_nums=params['split']
    # Get the split file paths
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((params['splits_path']).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
    else:
        split_files = []
        for s in split_nums:
            split_files.extend(list((params['splits_path']).glob(f"{source_data_name}_split_{s}_*.txt")))
    files_joined = [str(s) for s in split_files]

    print(f"Split id {split} out of {len(split_nums)} splits.")
    # Check that train, val, and test are available. Otherwise, continue to the next split.
    for phase in ["train", "val", "test"]:
        fname = build_split_fname(source_data_name, split, phase)
        if fname not in "\t".join(files_joined):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue

    for target_data_name in params['target_datasets']:
        ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"/ \
                f"split_{split}"
        if ml_data_dir.exists() is True:
            continue
        if params['only_cross_study'] and (source_data_name == target_data_name):
            continue # only cross-study
        print(f"\nSource data: {source_data_name}")
        print(f"Target data: {target_data_name}")

        params['ml_data_outdir'] = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
        frm.create_outdir(outdir=params["ml_data_outdir"])
        if source_data_name == target_data_name:
            # If source and target are the same, then infer on the test split
            test_split_file = f"{source_data_name}_split_{split}_test.txt"
        else:
            # If source and target are different, then infer on the entire target dataset
            test_split_file = f"{target_data_name}_all.txt"

        # Preprocess  data
        print("\nPreprocessing")
        train_split_file = f"{source_data_name}_split_{split}_train.txt"
        val_split_file = f"{source_data_name}_split_{split}_val.txt"
        print(f"train_split_file: {train_split_file}")
        print(f"val_split_file:   {val_split_file}")
        print(f"test_split_file:  {test_split_file}")
        print(f"ml_data_outdir:   {params['ml_data_outdir']}")
        if params['use_singularity']:
            raise Exception('Functionality using singularity is work in progress. Please use the Python version to call preprocess by setting use_singularity=False')

        else:
            preprocess_run = ["python",
                params['preprocess_python_script'],
                "--train_split_file", str(train_split_file),
                "--val_split_file", str(val_split_file),
                "--test_split_file", str(test_split_file),
                "--input_dir", params['input_dir'], 
                "--output_dir", str(ml_data_dir),
                "--y_col_name", str(params['y_col_name'])
            ]
            result = subprocess.run(preprocess_run, capture_output=True,
                                    text=True, check=True)
    return {'source_data_name':source_data_name, 'split':split}


@python_app 
def train(params, hp_model, source_data_name, split): 
    import subprocess
    hp = hp_model[source_data_name]
    if hp.__len__()==0:
        raise Exception(str('Hyperparameters are not defined for ' + source_data_name))
    
    model_dir = params['model_outdir'] / f"{source_data_name}" / f"split_{split}"
    ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{params['target_datasets'][0]}"/ \
                f"split_{split}"
    if model_dir.exists() is False:
        print("\nTrain")
        print(f"ml_data_dir: {ml_data_dir}")
        print(f"model_dir:   {model_dir}")
        if params['use_singularity']:
            raise Exception('Functionality using singularity is work in progress. Please use the Python version to call train by setting use_singularity=False')
        else:
            train_run = ["python", 
                         params['train_python_script'],
                        "--input_dir", str(ml_data_dir),
                        "--output_dir", str(model_dir),
                        "--epochs", str(params['epochs']),  # DL-specific
                        "--y_col_name", str(params['y_col_name']),
                        "--learning_rate", str(hp['learning_rate']),
                        "--batch_size", str(hp['batch_size'])
                    ]
            result = subprocess.run(train_run, capture_output=True,
                                    text=True, check=True)
    return {'source_data_name':source_data_name, 'split':split}

@python_app  
def infer(params, source_data_name, target_data_name, split): # 
    import subprocess
    model_dir = params['model_outdir'] / f"{source_data_name}" / f"split_{split}"
    ml_data_dir = params['ml_data_dir']/f"{source_data_name}-{target_data_name}"/ \
                f"split_{split}"
    infer_dir = params['infer_dir']/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    if params['use_singularity']:
        raise Exception('Functionality using singularity is work in progress. Please use the Python version to call infer by setting use_singularity=False')

    else:
        print("\nInfer")
        infer_run = ["python", params['infer_python_script'],
                "--input_data_dir", str(ml_data_dir),
                "--input_model_dir", str(model_dir),
                "--output_dir", str(infer_dir),
                "--y_col_name", str(params['y_col_name'])
        ]
        result = subprocess.run(infer_run, capture_output=True,
                                text=True, check=True)
    return True

###############################
####### CSA PARAMETERS ########
###############################

additional_definitions = CSA.additional_definitions
filepath = Path(__file__).resolve().parent
cfg = DRPPreprocessConfig() 
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.ini",
    default_model=None,
    additional_cli_section=None,
    additional_definitions=additional_definitions,
    required=None
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']
logger = logging.getLogger(f"{params['model_name']}")
params = frm.build_paths(params)  # paths to raw data

#Output directories for preprocess, train and infer
params['ml_data_dir'] = Path(params['output_dir']) / 'ml_data' 
params['model_outdir'] = Path(params['output_dir']) / 'models'
params['infer_dir'] = Path(params['output_dir']) / 'infer'

#Model scripts
params['preprocess_python_script'] = f"{params['model_name']}_preprocess_improve.py"
params['train_python_script'] = f"{params['model_name']}_train_improve.py"
params['infer_python_script'] = f"{params['model_name']}_infer_improve.py"

#Read Hyperparameters file
with open(params['hyperparameters_file']) as f:
    hp = json.load(f)
hp_model = hp[params['model_name']]

##########################################################################
##################### START PARSL PARALLEL EXECUTION #####################
##########################################################################

##Preprocess execution with Parsl
preprocess_futures=[]
for source_data_name in params['source_datasets']:
    for split in params['split']:
            preprocess_futures.append(preprocess(inputs=[params, source_data_name, split])) 

##Train execution with Parsl
train_futures=[]
for future_p in preprocess_futures:
    train_futures.append(train(params, hp_model, future_p.result()['source_data_name'], future_p.result()['split']))

##Infer execution with Parsl
infer_futures =[]
for future_t in train_futures:
    for target_data_name in params['target_datasets']:
        infer_futures.append(infer(params, future_t.result()['source_data_name'], target_data_name, future_t.result()['split']))

for future_i in infer_futures:
    print(future_i.result())

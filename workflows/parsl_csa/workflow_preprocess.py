import json
import logging
import sys
from pathlib import Path
from typing import Sequence, Tuple, Union

import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

import csa_params_def as CSA
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig

# Initialize parameters for CSA
additional_definitions = CSA.additional_definitions
filepath = Path(__file__).resolve().parent
cfg = DRPPreprocessConfig() 
params = cfg.initialize_parameters(
    pathToModelDir=filepath,
    default_config="csa_params.ini",
    additional_definitions=additional_definitions
)

##### CONFIG FOR LAMBDA ######
#available_accelerators: Union[int, Sequence[str]] = 8
worker_port_range: Tuple[int, int] = (10000, 20000)
retries: int = 1

config_lambda = Config(
    retries=retries,
    executors=[
        HighThroughputExecutor(
            address='127.0.0.1',
            label="htex_preprocess",
            cpu_affinity="alternating",
            #max_workers_per_node=2, ## IS NOT SUPPORTED IN Parsl version: 2023.06.19. CHECK HOW TO USE THIS???
            worker_debug=True,
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
def preprocess(inputs=[]):
    """ parsl implementation of preprocessing stage using python_app. """
    import json
    import subprocess
    import time
    import warnings
    from pathlib import Path

    def build_split_fname(source_data_name, split, phase):
        """ Build split file name. If file does not exist continue """
        if split=='all':
            return f"{source_data_name}_{split}.txt"
        return f"{source_data_name}_split_{split}_{phase}.txt"

    # python_app inputs
    params = inputs[0]
    source_data_name = inputs[1]
    split = inputs[2]

    split_nums = params['split']

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
            warnings.warn(f"\nThe {phase} split file {fname} is missing \
                          (continue to next split)")
            continue

    for target_data_name in params['target_datasets']:

        if params['only_cross_study'] and (source_data_name == target_data_name):
            continue # only cross-study
        print(f"\nSource data: {source_data_name}")
        print(f"Target data: {target_data_name}")

        ml_data_dir = params['ml_data_dir'] / \
            f"{source_data_name}-{target_data_name}" / f"split_{split}"
        if ml_data_dir.exists() is True:
            continue

        if source_data_name == target_data_name:
            # If source and target are the same, then infer on the test split
            test_split_file = f"{source_data_name}_split_{split}_test.txt"
        else:
            # If source and target are different, then infer on the entire
            # target dataset
            test_split_file = f"{target_data_name}_all.txt"

        # Preprocess  data
        print("\nPreprocessing")
        train_split_file = f"{source_data_name}_split_{split}_train.txt"
        val_split_file = f"{source_data_name}_split_{split}_val.txt"
        print(f"train_split_file: {train_split_file}")
        print(f"val_split_file:   {val_split_file}")
        print(f"test_split_file:  {test_split_file}")
        start = time.time()
        if params['use_singularity']:
            preprocess_run = ["singularity", "exec", "--nv",
                              params['singularity_image'], "preprocess.sh",
                              str("--train_split_file " + str(train_split_file)),
                              str("--val_split_file " + str(val_split_file)),
                              str("--test_split_file " + str(test_split_file)),
                              str("--input_dir " + params['input_dir']),
                              str("--output_dir " + str(ml_data_dir)),
                              str("--y_col_name " + str(params['y_col_name']))
            ]
        else:
            preprocess_run = ["bash", "execute_in_conda.sh",
                              params['model_environment'], 
                              params['preprocess_python_script'],
                              "--train_split_file", str(train_split_file),
                              "--val_split_file", str(val_split_file),
                              "--test_split_file", str(test_split_file),
                              "--input_dir", params['input_dir'], 
                              "--output_dir", str(ml_data_dir),
                              "--y_col_name", str(params['y_col_name'])
            ]

        result = subprocess.run(preprocess_run,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)

        # Logger
        print(f"returncode = {result.returncode}")
        result_file_name_stdout = ml_data_dir / 'logs.txt'
        with open(result_file_name_stdout, 'w') as file:
            file.write(result.stdout)

        # Timer
        time_diff = time.time() - start
        hours = int(time_diff // 3600)
        minutes = int((time_diff % 3600) // 60)
        seconds = time_diff % 60
        time_diff_dict = {'hours': hours,
                          'minutes': minutes,
                          'seconds': seconds}
        dir_to_save = ml_data_dir
        filename = 'runtime.json'
        with open(Path(dir_to_save) / filename, 'w') as json_file:
            json.dump(time_diff_dict, json_file, indent=4)

    return {'source_data_name': source_data_name, 'split': split}


###############################
####### CSA PARAMETERS ########
###############################

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']
logger = logging.getLogger(f"{params['model_name']}")

#Output directories for preprocess, train and infer
params['ml_data_dir'] = Path(params['output_dir']) / 'ml_data' 

#Model scripts
params['preprocess_python_script'] = f"{params['model_name']}_preprocess_improve.py"

##########################################################################
##################### START PARSL PARALLEL EXECUTION #####################
##########################################################################

##Preprocess execution with Parsl
preprocess_futures = []
for source_data_name in params['source_datasets']:
    for split in params['split']:
        preprocess_futures.append(
            preprocess(inputs=[params, source_data_name, split])
        ) 

for future_p in preprocess_futures:
    print(future_p.result())

parsl.dfk().cleanup()

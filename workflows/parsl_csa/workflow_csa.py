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
            label="htex",
            cpu_affinity="block",
            #max_workers_per_node=2, ## IS NOT SUPPORTED IN Parsl version: 2023.06.19. CHECK HOW TO USE THIS???
            worker_debug=True,
            worker_port_range=worker_port_range,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
            available_accelerators=params['available_accelerators'], 
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
def train(params, hp_model, source_data_name, split):
    """ parsl implementation of training stage using python_app. """
    import json
    import subprocess
    import time
    from pathlib import Path

    hp = hp_model[source_data_name]
    if hp.__len__() == 0:
        raise Exception(str('Hyperparameters are not defined for ' + source_data_name))
    
    model_dir = params['model_dir'] / f"{source_data_name}" / f"split_{split}"
    ml_data_dir = params['ml_data_dir'] / \
        f"{source_data_name}-{params['target_datasets'][0]}"/ f"split_{split}"

    if model_dir.exists() is False:
        print("\nTrain")
        print(f"ml_data_dir: {ml_data_dir}")
        print(f"model_dir:   {model_dir}")
        start = time.time()
        if params['use_singularity']:
            train_run = ["singularity", "exec", "--nv",
                        params['singularity_image'], "train.sh",
                        str("--input_dir " + str(ml_data_dir)),
                        str("--output_dir " + str(model_dir)),
                        str("--epochs " + str(params['epochs'])),
                        str("--y_col_name " + str(params['y_col_name'])),
                        str("--learning_rate " + str(hp['learning_rate'])),
                        str("--batch_size " + str(hp['batch_size']))
            ]
        else:
            train_run = ["bash", "execute_in_conda.sh",
                         params['model_environment'], 
                         params['train_python_script'],
                         "--input_dir", str(ml_data_dir),
                         "--output_dir", str(model_dir),
                         "--epochs", str(params['epochs']), # DL-specific
                         "--y_col_name", str(params['y_col_name']),
                         "--learning_rate", str(hp['learning_rate']),
                         "--batch_size", str(hp['batch_size'])
            ]

        result = subprocess.run(train_run,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)

        # Logger
        print(f"returncode = {result.returncode}")
        result_file_name_stdout = model_dir / 'logs.txt'
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
        dir_to_save = model_dir
        filename = 'runtime.json'
        with open(Path(dir_to_save) / filename, 'w') as json_file:
            json.dump(time_diff_dict, json_file, indent=4)

    return {'source_data_name': source_data_name, 'split': split}

@python_app  
def infer(params, source_data_name, target_data_name, split):
    """ parsl implementation of inferece stage using python_app. """
    import subprocess
    import json
    import time
    from pathlib import Path

    model_dir = params['model_dir'] / f"{source_data_name}" / f"split_{split}"
    ml_data_dir = params['ml_data_dir'] / \
        f"{source_data_name}-{target_data_name}" / f"split_{split}"
    infer_dir = params['infer_dir'] / \
        f"{source_data_name}-{target_data_name}" / f"split_{split}"

    print("\nInfer")
    start = time.time()
    if params['use_singularity']:
        infer_run = ["singularity", "exec", "--nv",
                    params['singularity_image'], "infer.sh",
                    str("--input_data_dir " + str(ml_data_dir)),
                    str("--input_model_dir " + str(model_dir)),
                    str("--output_dir " + str(infer_dir)),
                    str("--calc_infer_scores "+ "true"),
                    str("--y_col_name " + str(params['y_col_name']))
        ]
    else:
        infer_run = ["bash", "execute_in_conda.sh",
                     params['model_environment'], 
                     params['infer_python_script'],
                     "--input_data_dir", str(ml_data_dir),
                     "--input_model_dir", str(model_dir),
                     "--output_dir", str(infer_dir),
                     "--calc_infer_scores", "true",
                     "--y_col_name", str(params['y_col_name'])
        ]

    result = subprocess.run(infer_run,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)

    # Logger
    print(f"returncode = {result.returncode}")
    result_file_name_stdout = infer_dir / 'logs.txt'
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
    dir_to_save = infer_dir
    filename = 'runtime.json'
    with open(Path(dir_to_save) / filename, 'w') as json_file:
        json.dump(time_diff_dict, json_file, indent=4)

    return True

###############################
####### CSA PARAMETERS ########
###############################

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
fdir = Path(__file__).resolve().parent
y_col_name = params['y_col_name']
logger = logging.getLogger(f"{params['model_name']}")

#Output directories for preprocess, train and infer
params['ml_data_dir'] = Path(params['output_dir']) / 'ml_data' 
params['model_dir'] = Path(params['output_dir']) / 'models'
params['infer_dir'] = Path(params['output_dir']) / 'infer'

#Model scripts
params['train_python_script'] = f"{params['model_name']}_train_improve.py"
params['infer_python_script'] = f"{params['model_name']}_infer_improve.py"

#Read Hyperparameters file
with open(params['hyperparameters_file']) as f:
    hp = json.load(f)
hp_model = hp[params['model_name']]

##########################################################################
##################### START PARSL PARALLEL EXECUTION #####################
##########################################################################

##Train execution with Parsl
train_futures = []
for source_data_name in params['source_datasets']:
    for split in params['split']:
        train_futures.append(train(params, hp_model, source_data_name, split))

##Infer execution with Parsl
infer_futures = []
for future_t in train_futures:
    for target_data_name in params['target_datasets']:
        infer_futures.append(infer(params, future_t.result()['source_data_name'], target_data_name, future_t.result()['split']))

for future_i in infer_futures:
    print(future_i.result())

parsl.dfk().cleanup()

# Run HPO using DeepHyper on Lambda with conda

## 1. Install conda environment for the curated model 
Install model, IMPROVE, and datasets:
```
cd <WORKING_DIR>
git clone https://github.com/JDACS4C-IMPROVE/<MODEL>
cd <MODEL>
source setup_improve.sh
```

Install model environment (get the name of the yml file from model repo readme):
The workflow will need to know the ./<MODEL_ENV_NAME>/.
```
conda env create -f <MODEL_ENV>.yml -p ./<MODEL_ENV_NAME>/
```

## 2. Perform preprocessing
Run the preprocess script. 
The workflow will need to know the <PATH/TO/PREPROCESSED/DATA>.

```
cd PathDSP
conda activate ./<MODEL_ENV_NAME>/
python <MODEL_NAME>_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir <PATH/TO/PREPROCESSED/DATA>
conda deactivate
```

## 3. Install conda environment for DeepHyper
```
module load openmpi
conda create -n dh python=3.9 -y
conda activate dh
conda install gxx_linux-64 gcc_linux-64
pip install "deephyper[default]"
pip install mpi4py
```

## 4. Modify configuration file
`hpo_deephyper_params.ini` is an example configuration file for this workflow.
You will need to change the following parameters for your model:
`model_scripts_dir` should be set to the path to the model directory containing the model scripts (from step 1).
`input_dir` should be set to the location of the preprocessed data (above). We highly recommend that the name of this directory includes the source and split (e.g. ./ml_data/CCLE-CCLE/split_0). You can provide a complete or relative path, or the name of the directory if it is in `model_scripts_dir`.
`model_name` should be set to your model name (this should have the same capitalization pattern as your model scripts, e.g. deepttc for deepttc_preprocess_improve.py, etc).
`model_environment` should be set to the location of the model environment (from step 1). You can provide a complete or relative path, or the name of the directory if it is in `model_scripts_dir`.
`output_dir` should be set to path you would like the output to be saved to. We highly recommend that the name of this directory includes the source and split (e.g. ./deephyper/CCLE/split_0)
`epochs` should be set to the maximum number of epochs to train for.
`max_evals` should be set to the maximum number of evaluations to check for before launching additional training runs.
`interactive_session` should be set to True to run on Lambda, Polaris, and Biowulf. Other implementations have not yet been tested.
`hyperparameter_file` can be set to an alternate .json file containing hyperparameters. You can provide a complete or relative path, or the name of the directory if it is in `model_scripts_dir`. See below (step 5) for how to change hyperparameters.
`val_metric` can be set to any IMPROVE metric you would like to optimize. 'mse' and 'rmse' are minimized, all other metrics are maximized. Note that this does not change what val loss is used by the model, only what HPO tries to optimize. Default is 'mse'.
`num_gpus_per_node` should be set to the number of GPUs per node on your system. Default is 2.
Parameters beginning with `CBO_` can be used to change the optimization protocol. The names of these parameters can be found by running `python hpo_deephyper_subprocess.py --help` or looking in `hpo_deephyper_params_def.py`. Documentation of the DeepHyper CBO can be found here: https://deephyper.readthedocs.io/en/stable/_autosummary/deephyper.hpo.CBO.html#deephyper.hpo.CBO


## 5. Modify hyperparameters file
`hpo_deephyper_hyperparameters.json` contains dictionaries for the hyperparameters.
The default settings are as follows:

| Hyperparameter | Min  | Max  | Default |
| -------------- | ---- | ---- | ------- |
| batch_size     | 8    | 512  | 64      |
| learning_rate  | 1e-6 | 0.01 | 0.001   |

You can add more hyperparameters to test by adding additional dictionaries to this list. An example of an alternate hyperparameters file is `hpo_deephyper_hyperparameters_alternate.json`. Insure that the name is a valid parameter for the model you are using. Categorical hyperparameters can be added as follows:
```
    {
    "name": "early_stopping",
    "type": "categorical",
    "choices": [true, false], 
    "default": false
    }
```
Note that boolean values must be lowercase in JSON files.


## 6. Perform HPO
Navigate to the DeepHyper directory
```
cd <WORKING_DIR>/IMPROVE/workflows/deephyper_hpo
```
If necesssary (i.e not proceeding directly from above steps), activate environment:
```
module load openmpi 
conda activate dh
export PYTHONPATH=../../../IMPROVE
```

Run HPO:
```
mpirun -np 10 python hpo_deephyper_subprocess.py
```

To run HPO with a different config file:
```
mpirun -np 10 python hpo_deephyper_subprocess.py --config <ALTERNATE_CONFIG_FILE>
```


To submit a job on Polaris:
```
#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A IMPROVE_Aim1
#PBS -l filesystems=home:grand:eagle

module use /soft/modulefiles
module load nvhpc-mixed craype-accel-nvidia80
module load conda
conda activate

cd ${PBS_O_WORKDIR}

# MPI example w/ 4 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"


export PYTHONPATH=/lus/eagle/your/path/to/IMPROVE/

export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} python hpo_deephyper_subprocess.py
```




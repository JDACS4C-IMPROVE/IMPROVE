import json
import subprocess
import pandas as pd
import os
from pathlib import Path
import logging
import mpi4py
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import HpProblem, CBO
from mpi4py import MPI
import socket
import hpo_deephyper_params_def
from improvelib.initializer.config import Config

logging.basicConfig(
    # filename=f"deephyper.{rank}.log, # optional if we want to store the logs to disk
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

def locate_input(param_to_check, model_scripts_dir):
    checking = param_to_check
   # checks if param_to_check is a dir/file
    if not os.path.exists(param_to_check):
        # if param_to_check doesn't exist at that path, check if it's in model_scripts_dir
        param_to_check = os.path.join(model_scripts_dir,param_to_check)
        if not os.path.exists(param_to_check):
            print(f"Parameter {checking} provided but not found at provided path or in model_scripts_dir.") 
    return param_to_check

@profile
def run(job, optuna_trial=None):
    model_outdir_job_id = Path(params['output_dir'] + f"/{job.id}")
    train_run = ["bash", "hpo_deephyper_subprocess_train.sh",
             str(params['model_environment']),
             str(params['script_name']),
             str(params['input_dir']),
             str(model_outdir_job_id),
             str(os.environ["CUDA_VISIBLE_DEVICES"])
        ]
    if params['epochs'] is not None:
        train_run = train_run + ['epochs'] + [params['epochs']]
    for hp in params['hyperparams']:
        train_run = train_run + [str(hp)]
        train_run = train_run + [str(job.parameters[hp])]

    print(f"Launching run: ")
    print(train_run)
    subprocess_res = subprocess.run(train_run, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    # Logger
    print(f"returncode = {subprocess_res.returncode}")
    result_file_name_stdout = model_outdir_job_id / 'logs.txt'
    if model_outdir_job_id.exists() is False: # If subprocess fails, model_dir may not be created and we need to write the log files in model_dir
        os.makedirs(model_outdir_job_id, exist_ok=True)
    with open(result_file_name_stdout, 'w') as file:
        file.write(subprocess_res.stdout)

    # Load val_scores and get val_metric. Minimizes mse/rmse, maximizes all else.
    f = open(model_outdir_job_id / 'val_scores.json')
    val_scores = json.load(f)
    if params['val_metric'] in ('mse', 'rmse'):
        objective = -val_scores[params['val_metric']]
    elif params['val_metric'] in ('pcc', 'scc', 'r2', 'acc', 'recall', 'precision', 'f1', 'kappa', 'bacc', 'roc_auc', 'aupr'):
        objective = val_scores[params['val_metric']]

    # Checkpoint the model weights
    with open(f"{params['output_dir']}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
    # Initialize parameters for DeepHyper HPO
    filepath = Path(__file__).resolve().parent
    cfg = Config() 
    global params
    params = cfg.initialize_parameters(
        section="HPO",
        pathToModelDir=filepath,
        default_config="hpo_deephyper_params.ini",
        additional_definitions=hpo_deephyper_params_def.additional_definitions
    )
    output_dir = Path(params['output_dir'])
    if output_dir.exists() is False:
        os.makedirs(output_dir, exist_ok=True)

    # Configure parameters for DeepHyper HPO
    params['script_name'] = os.path.join(params['model_scripts_dir'],f"{params['model_name']}_train_improve.py")
    params['input_dir'] = locate_input(params['input_dir'], params['model_scripts_dir'])
    params['model_environment'] = locate_input(params['model_environment'], params['model_scripts_dir'])
    params['hyperparameter_file'] = locate_input(params['hyperparameter_file'], params['model_scripts_dir'])

    # Set hyperparameters
    problem = HpProblem()
    with open(params['hyperparameter_file']) as f:
        hyperparams = json.load(f)
    for hp in hyperparams:
        if hp['type'] == "categorical":
            problem.add_hyperparameter(hp['choices'], hp['name'], default_value=hp['default'])
        else:
            if hp['log_uniform']:
                problem.add_hyperparameter((hp['min'], hp['max'], "log-uniform"), 
                                        hp['name'], default_value=hp['default'])
            else:
                problem.add_hyperparameter((hp['min'], hp['max']), 
                                        hp['name'], default_value=hp['default'])
    params['hyperparams'] = [d['name'] for d in hyperparams]

    # Enable using multiple GPUs
    mpi4py.rc.initialize = False
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "multiple"
    mpi4py.rc.recv_mprobe = False
    if not MPI.Is_initialized():
        MPI.Init_thread()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % params['num_gpus_per_node'])
    cuda_name = "cuda:" + str(rank % params['num_gpus_per_node'])

    # Run DeepHyper
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            print(problem)
            search = CBO(
                problem,
                evaluator,
                log_dir=params['output_dir'],
                verbose=1,
                surrogate_model = params['CBO_surrogate_model'],
                acq_func = params['CBO_acq_func'],
                acq_optimizer = params['CBO_acq_optimizer'],
                acq_optimizer_freq = params['CBO_acq_optimizer_freq'],
                kappa = params['CBO_kappa'],
                xi = params['CBO_xi'],
                update_prior = params['CBO_update_prior'],
                update_prior_quantile = params['CBO_update_prior_quantile'],
                n_jobs = params['CBO_n_jobs'],
                n_initial_points = params['CBO_n_initial_points'],
                initial_point_generator = params['CBO_initial_point_generator'],
                filter_failures = params['CBO_filter_failures'],
                max_failures = params['CBO_max_failures'],
            )
            results = search.search(max_evals=params['max_evals'])
            results = results.sort_values(f"m:{params['val_metric']}", ascending=True)
            results.to_csv(f"{params['output_dir']}/hpo_results.csv", index=False)
    print("current node: ", socket.gethostname(), "; current rank: ", rank, "; CUDA_VISIBLE_DEVICE is set to: ", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Finished deephyper HPO.")

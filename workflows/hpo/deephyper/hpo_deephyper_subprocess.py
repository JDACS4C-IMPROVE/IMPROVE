import json
import logging
import os
import socket
import subprocess
from pathlib import Path

import mpi4py
from mpi4py import MPI
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

import hpo_deephyper_params_def
from improvelib.initializer.config import Config

# Set up logging
log_level = os.getenv("IMPROVE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

# Set up logger for this module
logger = logging.getLogger(__name__)


def locate_input(param_to_check, model_scripts_dir):
    """
    This func assists in verifying that some essential parameters specified in the
    config file actually exist. This includes parameters such as:
    - input_dir (which should contain the preprocessed data)
    - model_environment (which is activated for the model in the subprocess)
    - and hyperparameter_file (which contains the hyperparameters and ranges)

    If a parameter is not found, the function checks if it's in model_scripts_dir.
    If it's not, the function prints a warning.
    """
    # TODO: should this be an error? if yes, is it a problem if it's run in parallel?
    checking = param_to_check
    # checks if param_to_check is a dir/file
    if not os.path.exists(param_to_check):
        # if param_to_check doesn't exist at that path, check if it's in model_scripts_dir
        param_to_check = os.path.join(model_scripts_dir, param_to_check)
        if not os.path.exists(param_to_check):
            logger.warning(f"Parameter {checking} provided but not found at provided path or in model_scripts_dir.") 
    return param_to_check


@profile
def run(job):
    """Run a single training job and return the optimization objective.

    Args:
        job: DeepHyper job object containing hyperparameters and job ID

    Returns:
        dict: Contains 'objective' (float) and 'metadata' (dict of validation scores)

    Process:
        1. Builds bash command with hyperparameters
        2. Executes training subprocess and captures stdout/stderr
        3. Writes subprocess output to <output_dir>/<job.id>/logs.txt
        4. Loads validation scores from <job.id>/val_scores.json
        5. Computes objective (negated for mse/rmse, direct for others)
        6. Returns objective and raw validation scores
    """
    model_outdir_job_id = Path(params['output_dir'] + f"/{job.id}")
    train_run = [
        "bash",
        "hpo_deephyper_subprocess_train.sh",
        str(params['model_environment']), # Conda env
        str(params['script_name']), # e.g., graphdrp_train_improve.py
        str(params['input_dir']),
        str(model_outdir_job_id),
        str(os.environ["CUDA_VISIBLE_DEVICES"]),
    ]
    if params['epochs'] is not None:
        train_run.extend(['epochs', str(params['epochs'])])
    for hp in params['hyperparams']:
        train_run.extend([str(hp), str(job.parameters[hp])])

    logger.info("Launching run:")
    logger.info(f"Command: {' '.join(train_run)}")
    subprocess_res = subprocess.run(
        train_run,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Logger
    logger.info(f"Training subprocess return code: {subprocess_res.returncode}")
    result_file_name_stdout = model_outdir_job_id / 'logs.txt'
    # If subprocess fails, model_outdir_job_id may not be created. If it's not,
    # we create it and then write the logs into result_file_name_stdout
    if model_outdir_job_id.exists() is False:
        os.makedirs(model_outdir_job_id, exist_ok=True)
    with open(result_file_name_stdout, 'w') as file:
        file.write(subprocess_res.stdout)

    # Load val_scores, get val_metric, and set objective. Minimizes mse/rmse, maximizes all else.
    with open(model_outdir_job_id / 'val_scores.json') as val_file:
        val_scores = json.load(val_file)
    
    val_metric = params['val_metric']
    if val_metric in ('mse', 'rmse'):
        objective = -val_scores[val_metric]
    elif val_metric in (
        'pcc', 'scc', 'r2', 'acc', 'recall', 'precision', 'f1', 'kappa',
        'bacc', 'roc_auc', 'aupr'
    ):
        objective = val_scores[val_metric]

    logger.info(f"Job {job.id} completed - {val_metric}: {val_scores[val_metric]:.6f}, objective: {objective:.6f}")

    # # Checkpoint the model weights
    # with open(f"{params['output_dir']}/model_{job.id}.pkl", "w") as f:
    #     f.write("model weights")

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
    # script_name is the script to execute in run(); for HPO is usually the train script
    params['script_name'] = os.path.join(params['model_scripts_dir'], f"{params['model_name']}_train_improve.py")
    params['input_dir'] = locate_input(params['input_dir'], model_scripts_dir=params['model_scripts_dir'])
    params['model_environment'] = locate_input(params['model_environment'], model_scripts_dir=params['model_scripts_dir'])
    params['hyperparameter_file'] = locate_input(params['hyperparameter_file'], model_scripts_dir=params['model_scripts_dir'])

    # Set hyperparameters
    problem = HpProblem()
    with open(params['hyperparameter_file']) as f:
        hyperparams = json.load(f)
    for hp in hyperparams:
        if hp['type'] == "categorical":
            problem.add_hyperparameter(
                value=hp['choices'],
                name=hp['name'],
                default_value=hp['default']
            )
        else:
            if hp['log_uniform']:
                problem.add_hyperparameter(
                    value=(hp['min'], hp['max'], "log-uniform"),
                    name=hp['name'],
                    default_value=hp['default']
                )
            else:
                problem.add_hyperparameter(
                    value=(hp['min'], hp['max']),
                    name=hp['name'],
                    default_value=hp['default']
                )
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
    
    # Add per-rank log file (useful for Polaris)
    log_file = f"{params['output_dir']}/deephyper_rank_{rank}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s"
    ))
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    # Run DeepHyper
    # Use method="serial" to step through the code:
    # python -m pdb hpo_deephyper_subprocess.py --config hpo_deephyper_params.ini
    with Evaluator.create(
        run,
        method="mpicomm",
        # method="serial", # allows to step through the code
        method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            logger.info(f"HPO problem definition: {problem}")
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
            logger.info(f"Starting HPO search with {params['max_evals']} evaluations")
            results = search.search(max_evals=params['max_evals'])
            results = results.sort_values(f"m:{params['val_metric']}", ascending=True)
            results.to_csv(f"{params['output_dir']}/hpo_results.csv", index=False)
            logger.info(f"HPO search completed. Results saved to {params['output_dir']}/hpo_results.csv")
            best_metric_col = f"m:{params['val_metric']}"
            logger.info(f"Best {params['val_metric']}: {results.iloc[0][best_metric_col]:.6f}")

    logger.info(f"Node: {socket.gethostname()}, Rank: {rank}, CUDA device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info("Finished DeepHyper HPO.")

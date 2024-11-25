"""
Before running this script, first need to preprocess the data.

It is assumed that the csa benchmark data is downloaded via download_csa.sh
and the env vars $IMPROVE_DATA_DIR and $PYTHONPATH are set:
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/path/to/IMPROVE_lib

mpirun -np 10 python hpo_subprocess.py
"""
# import copy
import json
import subprocess
import pandas as pd
import os
import logging
import mpi4py
from mpi4py import MPI
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
import socket

# ---------------------
# Enable using multiple GPUs
# ---------------------

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_rank = os.environ["PMI_LOCAL_RANK"]

# CUDA_VISIBLE_DEVICES is now set via set_affinity_gpu_polaris.sh
# uncomment the below commands if running via interactive node
#num_gpus_per_node = 3
#os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus_per_node)

# ---------------------
# Enable logging
# ---------------------

logging.basicConfig(
    # filename=f"deephyper.{rank}.log, # optional if we want to store the logs to disk
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    force=True,
)

# ---------------------
# Hyperparameters
# ---------------------
problem = HpProblem()

problem.add_hyperparameter((8, 512, "log-uniform"), "batch_size", default_value=8)
problem.add_hyperparameter((1e-6, 1e-2, "log-uniform"),
                           "learning_rate", default_value=0.0004)
problem.add_hyperparameter((0, 0.5), "dropout", default_value=0.1)
# problem.add_hyperparameter([True, False], "early_stopping", default_value=False)

# ---------------------
# Some IMPROVE settings
# ---------------------
source = "GDSCv1"
split = 0
train_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
val_ml_data_dir = f"ml_data/{source}-{source}/split_{split}"
model_outdir = f"dh_hpo_improve/{source}/split_{split}"
log_dir = "dh_hpo_logs/"
subprocess_bashscript = "subprocess_train_singularity.sh"


@profile
def run(job, optuna_trial=None):

    # config = copy.deepcopy(job.parameters)
    # params = {
    #     "epochs": DEEPHYPER_BENCHMARK_MAX_EPOCHS,
    #     "timeout": DEEPHYPER_BENCHMARK_TIMEOUT,
    #     "verbose": False,
    # }
    # if len(config) > 0:
    #     remap_hyperparameters(config)
    #     params.update(config)

    model_outdir_job_id = model_outdir + f"/{job.id}"
    learning_rate = job.parameters["learning_rate"]
    batch_size = job.parameters["batch_size"]
    dropout = job.parameters["dropout"]
    # val_scores = main_train_grapdrp([
    #     "--train_ml_data_dir", str(train_ml_data_dir),
    #     "--val_ml_data_dir", str(val_ml_data_dir),
    #     "--model_outdir", str(model_outdir_job_id),
    # ])
    subprocess_res = subprocess.run(
        [
            "bash", subprocess_bashscript,
             str(train_ml_data_dir),
             str(val_ml_data_dir),
             str(model_outdir_job_id),
             str(learning_rate),
             str(batch_size),
             str(dropout),
             str(os.environ["CUDA_VISIBLE_DEVICES"])
        ], 
        capture_output=True, text=True, check=True
    )
    
    # print(subprocess_res.stdout)
    # print(subprocess_res.stderr)

    # Load val_scores and get val_loss
    # f = open(model_outdir + "/val_scores.json")
    f = open(os.path.join(os.environ["IMPROVE_DATA_DIR"], model_outdir_job_id, "val_scores.json"))
    val_scores = json.load(f)
    objective = -val_scores["val_loss"]
    # print("objective:", objective)

    # Checkpoint the model weights
    with open(f"{log_dir}/model_{job.id}.pkl", "w") as f:
        f.write("model weights")

    # return score
    return {"objective": objective, "metadata": val_scores}


if __name__ == "__main__":
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:

        if evaluator is not None:
            print(problem)

            search = CBO(
                problem,
                evaluator,
                log_dir=log_dir,
                verbose=1,
            )

            # max_evals = 2
            # max_evals = 4
            # max_evals = 10
            # max_evals = 20
            max_evals = 100
            # max_evals = 100
            results = search.search(max_evals=max_evals)
            results = results.sort_values("m:val_loss", ascending=True)
            results.to_csv(os.path.join(os.environ["IMPROVE_DATA_DIR"], model_outdir, "hpo_results.csv"), index=False)
    print("current node: ", socket.gethostname(), "; current rank: ", rank, "; local rank", local_rank, "; CUDA_VISIBLE_DEVICE is set to: ", os.environ["CUDA_VISIBLE_DEVICES"])
    print("Finished deephyper HPO.")

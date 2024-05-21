import parsl
from parsl import python_app , bash_app

from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher # USE the MPIExecLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints


import parsl
from parsl import python_app, bash_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
# from parsl.config import Config
from parsl.executors import HighThroughputExecutor


from Workflows import Demo
from CLI import CLI

from Config.Parsl import Config as Parsl
from Config.CSA import Config as CSA



# Adjust your user-specific options here:
run_dir="~/tmp"


print(parsl.__version__)

user_opts = {
    "worker_init":      f"source ~/.venv/parsl/bin/activate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand -l singularity_fakeroot=true" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE",
    "queue":            "R1819593",
    "walltime":         "1:00:00",
    "nodes_per_block":  10, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
}



def get_config():
    config = None

    if False:

        config = Config(
                executors=[
                    HighThroughputExecutor(
                        label="htex",
                        available_accelerators=4, # if this is set, it will override other settings for max_workers if set
                        max_workers_per_node=4, # Set as many workers as there are GPUs because we want one worker to use 1 GPU
                        address=address_by_interface("bond0"),
                        cpu_affinity="block-reverse",
                        prefetch_capacity=0,
                        worker_debug=True,
                        # start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
                        provider=PBSProProvider(
                            launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
                            account=user_opts["account"],
                            queue=user_opts["queue"],
                            select_options="ngpus=4",
                            # PBS directives (header lines): for array jobs pass '-J' option
                            scheduler_options=user_opts["scheduler_options"],
                            # Command to be run before starting a worker, such as:
                            worker_init=user_opts["worker_init"],
                            # number of compute nodes allocated for each block
                            nodes_per_block=user_opts["nodes_per_block"],
                            init_blocks=1,
                            min_blocks=0,
                            max_blocks=1, # Can increase more to have more parallel jobs
                            # cpus_per_node=user_opts["cpus_per_node"],
                            walltime=user_opts["walltime"]
                        ),
                    ),
                ],
                retries=2,
                app_cache=True,
        )
    else:
        # Local Config
        config = Config(
        executors=[
            HighThroughputExecutor(
                label="hpo_local",
                worker_debug=True,
                cores_per_worker=1,
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=1,
                    max_blocks=1,
                ),
                max_workers_per_node=10,
            )
        ],
        strategy=None,
    )



additional_definitions = {
    "model" : {
        "type" : str,
        "default" : None,
        "help" : "Singulritiy image for the model",
        "choices" : None,
        "nargs" : None
    },
    "data_set" : {  
        "type" : str,
        "default" : None,
        "help" : "Data set",
        "choices" : None,
        "nargs" : None
    },
}

cli = CLI()
cli.set_command_line_options(options=additional_definitions)
cli.get_command_line_options()

pcfg = Parsl()
pcfg = parsl_config.load_config(cli.params['parsl_config_file'])

csa = CSA()
csa = csa.load_config(cli.params['csa_config_file'])


futures = {}
parsl.clear()
# checkpoints = get_all_checkpoints(run_dir)
# print("Found the following checkpoints: ", checkpoints)
with parsl.load(pcfg.config):

    results = Demo.run(config={},debug=True)

    for key in results.keys():
        print(f"{key} : {results[key]}")


    parsl.clear()
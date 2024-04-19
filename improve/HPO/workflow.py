#! /usr/bin/env python

import parsl
from parsl import python_app, bash_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor



from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface
# For checkpointing:
from parsl.utils import get_all_checkpoints

# Adjust your user-specific options here:
run_dir="/lus/grand/projects/yourproject/yourrundir/"
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher # USE the MPIExecLauncher
# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface


user_opts = {
    "worker_init":      f"source /path/to/your/virtualenv/bin/activate; cd {run_dir}", # load the environment where parsl is installed
    "scheduler_options":"#PBS -l filesystems=home:eagle:grand" , # specify any PBS options here, like filesystems
    "account":          "IMPROVE",
    "queue":            "debug-scaling",
    "walltime":         "1:00:00",
    "nodes_per_block":  3, # think of a block as one job on polaris, so to run on the main queues, set this >= 10
}

checkpoints = get_all_checkpoints(run_dir)
print("Found the following checkpoints: ", checkpoints)

# config = Config(
#         executors=[
#             HighThroughputExecutor(
#                 label="hpo_polaris",
#                 available_accelerators=4, # if this is set, it will override other settings for max_workers if set
#                 max_workers_per_node=4, # Set as many workers as there are GPUs because we want one worker to use 1 GPU
#                 address=address_by_interface("bond0"),
#                 cpu_affinity="block-reverse",
#                 prefetch_capacity=0,
#                 start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
#                 provider=PBSProProvider(
#                     launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
#                     account=user_opts["account"],
#                     queue=user_opts["queue"],
#                     select_options="ngpus=4",
#                     # PBS directives (header lines): for array jobs pass '-J' option
#                     scheduler_options=user_opts["scheduler_options"],
#                     # Command to be run before starting a worker, such as:
#                     worker_init=user_opts["worker_init"],
#                     # number of compute nodes allocated for each block
#                     nodes_per_block=user_opts["nodes_per_block"],
#                     init_blocks=1,
#                     min_blocks=0,
#                     max_blocks=1, # Can increase more to have more parallel jobs
#                     cpus_per_node=user_opts["cpus_per_node"],
#                     walltime=user_opts["walltime"]
#                 ),
#             ),
#         ],
#         retries=2,
#         app_cache=True,
# )


# Local Config
local_config = Config(
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
            # max_workers_per_node=10,
        )
    ],
    strategy=None,
)


# parsl.clear()
#parsl.load(local_threads)


@python_app
def hello_python (message):
    import time
    time.sleep(1)
    return 'Hello %s' % message

@bash_app
def hello_bash(message, stdout='hello-stdout'):
    return 'echo "Hello %s" ; env ; sleep 5' % message


with parsl.load():
    # invoke the Python app and print the result
    print(hello_python('World (Python)').result())
    hello_bash("World (Bash)").result()
    # invoke the Bash app and read the result from a file
    out = []
    for i in range(5):
        print(i)
        # hello_bash("World " + str(i) + " (Bash)").result()
        print(hello_python('World (Python)').result())

print('Bash app wrote to hello-stdout:')
with open('hello-stdout', 'r') as f:
    print(f.read())
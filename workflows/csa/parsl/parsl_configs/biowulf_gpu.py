from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.addresses import address_by_interface
from parsl.launchers import SrunLauncher

def get_parsl_config(available_accelerators = None):
    parsl_config = Config(
        executors=[
            HighThroughputExecutor(
                label="biowulf_htex",
                address=address_by_interface("ib0"),
                available_accelerators=4,
                provider=SlurmProvider(
                    nodes_per_block=1,
                    cores_per_node=56,
                    mem_per_node=120,
                    exclusive=True,
                    walltime="4:00:00",
                    init_blocks=1,
                    max_blocks=2,
                    partition="gpu",
                    scheduler_options="#SBATCH --gres=gpu:k80:4",
                    regex_job_id=r"(?P<id>^\d+$)",
                    launcher=SrunLauncher(),
                ),
            )
        ],
    )
    return parsl_config
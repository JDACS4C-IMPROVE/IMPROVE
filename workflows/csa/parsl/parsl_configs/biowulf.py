from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

def get_parsl_config(available_accelerators = None):
    parsl_config = Config(
        executors=[
            HighThroughputExecutor(
                address = "",
                label="htex_local",
                worker_debug=True,
                cores_per_worker=1,
                max_workers_per_node=6,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
    )
    return parsl_config
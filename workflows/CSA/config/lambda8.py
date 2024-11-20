from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider


parsl_config = Config(
    retries=1,
    executors=[
        HighThroughputExecutor(
            address='127.0.0.1',
            label="htex",
            cpu_affinity="block",
            max_workers_per_node=6, ## IS NOT SUPPORTED IN  Parsl version: 2023.06.19. CHECK HOW TO USE THIS???
            worker_debug=True,
            available_accelerators=6 ,  ## CHANGE THIS AS REQUIRED BY THE MACHINE
            # worker_port_range=worker_port_range,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy='simple',
)

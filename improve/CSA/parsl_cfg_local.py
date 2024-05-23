import parsl

from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor


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

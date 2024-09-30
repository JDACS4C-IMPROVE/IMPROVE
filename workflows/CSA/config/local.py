from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

parsl_config = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=8,
            label='local_threads'
        )
    ]
)
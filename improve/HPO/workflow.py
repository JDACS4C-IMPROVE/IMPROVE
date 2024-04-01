#! /usr/bin/env python

import parsl
from parsl import python_app, bash_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

local_htex = Config(
    executors=[
        HighThroughputExecutor(
            label="htex_Local",
            worker_debug=True,
            cores_per_worker=1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy=None,
)

parsl.clear()
#parsl.load(local_threads)


@python_app
def hello_python (message):
    return 'Hello %s' % message

@bash_app
def hello_bash(message, stdout='hello-stdout'):
    return 'echo "Hello %s"' % message


with parsl.load(local_htex):
    # invoke the Python app and print the result
    print(hello_python('World (Python)').result())

    # invoke the Bash app and read the result from a file
    hello_bash('World (Bash)').result()

print('Bash app wrote to hello-stdout:')
with open('hello-stdout', 'r') as f:
    print(f.read())
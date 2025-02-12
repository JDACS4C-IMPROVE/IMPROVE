## Requirements

IMPROVE general env
An IMPROVE compliant model
A system with swarm


## Install model

Requires conda env to be installed in the model repo location

Same parameters as bruteforce for now.

Get swarm files

Activate general IMPROVE env
export PYTHONPATH

```
python lca_swarm.py
```


Currently has PYTHONPATH hardcoded in as four levels up, should be unnecessary with pip install of improvelib in the models

Assumes conda env is located in the model dir, which can be anywhere as long as it is specified in the configs

Currently just writes swarm files to pwd because output dir is used for the output of the swarm files.

## Running swarm files

Example usage for Biowulf

```
swarm --merge-output -g 30 --time-per-command 00:10:00 -J model_preprocess preprocess.swarm
```

```
swarm --merge-output --partition=gpu --gres=gpu:k80:1 -g 60 --time-per-command 06:00:00 -J model_train train.swarm
```

```
swarm --merge-output --partition=gpu --gres=gpu:k80:1 -g 60 --time-per-command 00:30:00 -J model_train infer.swarm
```

You may need to change the memory (`-g`) and time (`--time-per-command`) allocations for your model. See Biowulf documentation for Swarm here: https://hpc.nih.gov/apps/swarm.html

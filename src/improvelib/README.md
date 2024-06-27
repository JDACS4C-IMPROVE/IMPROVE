# Package improvelib

This is the top level directory for the improve framework code governing the configuration and command line user experience for data preprocessing, model training and model execution (inference). The goal is to provide a base set of common configuration options, command line parameters and data processing methods for all model scripts. 

The top level directory contains utility functions and parent classes providing base functionality for handling config and CLI. Specialized classes for preprocessing, training and inference inherit from the parent classes.


## Content

.
├── Apps
│   └── DrugResponsePrediction
│       └── common_parameters.yml
├── Benchmarks
│   ├── Base.py
│   └── DrugResponsePrediction.py
├── cli.py
├── config.py
├── csa.py
├── data_loader.py
├── infer.py
├── metrics.py
└── Tests
    └── Data
        ├── common_parameters.yml
        └── default.cfg







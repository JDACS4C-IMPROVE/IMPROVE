# Unit Tests 

This directory contains unit test files for our project. Each file corresponds to a specific module or functionality in our main codebase. The purpose of these tests is to ensure that all functions behave as expected under a variety of conditions.

## Set Up

For setting up the environment to test `improvelib`, first clone the `IMPROVE` repo:

```bash
https://github.com/JDACS4C-IMPROVE/IMPROVE.git
```

Two options for testing `improvelib` are:
1. Set the path to the repo as an environment variable and install the dependencies (see [README.md](../README.md)).
```bash
export MY_PATH_TO_IMPROVE=`pwd`
export PYTHONPATH=$PYTHONPATH:${MY_PATH_TO_IMPROVE}
```
2. Use pip to install the `improvelib` package.
```bash
pip install improvelib
```

## Running the Tests

To run all the tests, navigate to the `tests` folder and run the following command:

```bash
python -m unittest
```

To run an individual test, run the command below:
```bash
python test_params.py 
```

## Adding a New Test

`improvelib` uses the Python package `unittest`. To add a new test, name your file starting with `test_*.py`.

At the top of this file, include a section describing the purpose of the unit test. 

The end of the file must contain the following:

```bash
if __name__ == "__main__":
    unittest.main()
```
For an example of a unit test file, see `test_drp_params.py`.

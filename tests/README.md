# Unit Tests 

This directory contains unit test files for our project. Each file corresponds to a specific module or functionality in our main codebase. The purpose of these tests is to ensure that all functions behave as expected under a variety of conditions.

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
For an example of a unit test file, see `test_params.py`.

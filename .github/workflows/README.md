# IMPROVE UnitTest GitHub Actions (unittest-actions.yml)

This repository contains a GitHub Actions workflow to run unit tests for the IMPROVE library. The workflow is triggered on every push to the repository.

The workflow is defined in the `.github/workflows/unittest-actions.yml` file. Here are the steps it performs:

1. **Checkout Code**: The repository code is checked out using the `actions/checkout` action.
2. **Install and Check**: The workflow installs the IMPROVE library in editable mode and verifies the installation by printing the version. It then sets the `PYTHONPATH` environment variable and creates necessary directories for data. Finally, it runs the `test_drp_params.py` test script.

# Docker Parsl GraphDRP CSA Workflow (docker_gdrp_parsl.yml)

This GitHub Actions workflow is designed to automate the testing and execution of the GraphDRP CSA (Cross-Species Analysis) workflow using Docker. The workflow performs the following steps:

1. **Checkout the Repository**: The workflow checks out the IMPROVE repository code.
2. **Install Missing Libraries**: Installs necessary system libraries such as `libxrender1`, `libxext6`, and `tree`.
3. **Install and Check**: Verifies the Python and Conda environments, activates the `GraphDRP` Conda environment (already has csa-data downloaded), and installs the required Python packages.
4. **Run Preprocessing and CSA Workflows**: Executes the preprocessing and Parsl CSA workflows using the specified configuration files.
## Configuration Files

The workflow uses the following configuration files:

- **`parsl_csa_githubactions.ini`**: This configuration file is used to set up the parameters for the Parsl CSA workflow. It includes settings for data paths, model parameters, and other relevant configurations.

# Docker GraphDRP LCA Workflow (docker_gdrp_lca.yml)

This GitHub Actions workflow automates the execution of the GraphDRP Learning Curve Analysis (LCA) workflow using Docker. The workflow performs the following steps:

1.  **Checkout the Repository**: The workflow checks out the IMPROVE repository code.
2.  **Install Missing Libraries and check**
3.  **Generate LCA Splits**: Executes the `gen_lc_splits.sh` script to create learning curve data splits.
4.  **Run Bruteforce LCA Workflow**: Executes the `lca_bruteforce.py` script using the specified configuration file.
5.  **Run Postprocessing**: Executes the `lca_postprocess.py` script to process the results of the LCA workflow.

## Configuration Files

The workflow uses the following configuration file:

- **`lca_bruteforce_githubactions.ini`**: Located in `workflows/lca/bruteforce/`, this file contains parameters for the bruteforce LCA workflow, including settings for data paths and model configurations.



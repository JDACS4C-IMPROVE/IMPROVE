import sys
import os
import importlib
import parsl
from pathlib import Path
from parsl import bash_app

def check_model_script(model_dir, model_name, stage):
    """Check if the model script is valid."""
    # Create model script name
    model_preprocess_path = Path(model_dir) / f"{model_name}_{stage}_improve.py"
    # Check if model script exists
    if os.path.isfile(model_preprocess_path):
        # Make absolute path
        script = os.path.abspath(model_preprocess_path)
    else:
        raise FileNotFoundError(f"Script {model_preprocess_path} does not exist.")
    return script

def _load_parsl_config(parsl_config_file, available_accelerators):
    # if the config file is a yaml file load it as yaml
    if parsl_config_file.endswith('.py'):
        # import the parsl_config from the file
        # add the directory of the config file to the path
        sys.path.append(os.path.dirname(parsl_config_file))
        # get filename without extension
        filename = os.path.basename(parsl_config_file).split('.')[0]
        # import the file
        imported_parsl_config = importlib.import_module(filename)
        # assign the parsl_config to self.parsl_config
        parsl_config = imported_parsl_config.get_parsl_config(available_accelerators)
    else:
        raise ValueError("Unknown file format for parsl_config_file")
    return parsl_config

def init_parsl(parsl_config_file, available_accelerators):
    """Initialize the Parsl configuration."""
    parsl_config = _load_parsl_config(parsl_config_file, available_accelerators)
    # Disable logging for Parsl
    parsl_config.initialize_logging = False
    parsl_config.usage_tracking = True
    # Load the Parsl configuration
    parsl.clear()
    parsl.load(parsl_config)
    return

def shutdown_parsl():
    """Shutdown the Parsl configuration."""
    parsl.dfk().cleanup()
    parsl.clear()
    return


@bash_app
def make_call(script_call = None, conda_env = None, stderr = "stderr.txt", stdout = "stdout.txt", inputs = [], outputs = []):
    """Preprocess the input file using the script."""
    import logging 
    logger = logging.getLogger(__name__)
    # Prefix and activate the conda environment
    prefix = f"START=$(date +%s) ; echo Start:\t$START ; conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {conda_env} ; "
    SUFFIX=' ; STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1'
    call = prefix + script_call + SUFFIX
    logger.debug(f"Preprocessing command: {call}")
    return call
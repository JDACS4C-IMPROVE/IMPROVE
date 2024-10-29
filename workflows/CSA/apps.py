import os
import logging
from typing import Sequence, Tuple, Union
from parsl.data_provider.files import File
from parsl.app.app import python_app, bash_app

logger = logging.getLogger(__name__)
logger.setLevel( os.getenv("IMPROVE_LOG_LEVEL", logging.INFO))

prefix= "START=$(date +%s) ; echo Start:\t$START "
suffix= "STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1"

def get_conda_env(conda_env):
    if conda_env:
        conda= f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {conda_env} "
    else:
        conda = ""
    return conda

def make_call(cli, conda_env):

    call=[]

    if prefix:
        call.append(prefix)
    if conda_env:
        call.append(get_conda_env(conda_env))
    call.append(" ".join(cli))
    if suffix:
        call.append(suffix)

    return " ;".join(call)



@bash_app
def preprocess( 
    input_dir = None,
    output_dir = None,
    train_file = None, 
    val_file = None,
    test_file = None,
    column_name = None,
    conda_env = None,
    script = None,
    
    stderr = "stderr.txt",
    stdout = "stdout.txt",
    inputs = [],
    outputs = [], # File(os.path.join(os.getcwd(), 'my_stdout*'))
    ):
    """Preprocess the input file using the script."""
    
    logger.info(f"Setting up preprocess: {script} on input data {input_dir} and saving to {output_dir}")

    # Create the command line interface for preprocessing
    cli = [ "time",
           script,
        "--train_split_file" , train_file,
        "--val_split_file" , val_file,
        "--test_split_file" , test_file,
        "--input_dir" , input_dir,
        "--output_dir" , output_dir,
        "--y_col_name" , column_name
    ]

    call = " ".join(cli)

    logger.debug(f"Preprocessing command: {call}")
    return call


@bash_app
def train( 
    script: str = None, 
    input_dir: str = None, 
    output_dir: str = None, 
    epochs: int = 100, 
    column_name: str = None, 
    learning_rate: float = 0.001, 
    batch_size: int = 32 , 
    conda_env: str = None,
    stdout: str = "stdout.txt", 
    stderr: str = "stderr.txt", 
    inputs: Sequence[File] = [], 
    outputs: Sequence[File] = [] ):    
    
    logger.info(f"Setting up train: {script} on input data {input_dir} and saving to {output_dir}")

    cli = [ "time",
           str(script), 
           "--input_dir" , input_dir,
           "--output_dir" , output_dir,
        #    "--epochs" , epochs,
        #    "--y_col_name" , column_name,
        #    "--learning_rate" , learning_rate,
        #    "--batch_size" , batch_size
           ]
    
    cli.append(f"--epochs {epochs}") if epochs else None
    cli.append(f"--y_col_name {column_name}") if column_name else None
    cli.append(f"--learning_rate {learning_rate}") if learning_rate else None
    cli.append(f"--batch_size {batch_size}") if batch_size else None

    call = make_call(cli, conda_env)
    logger.debug(f"Train call: {call}")

    return call 


@bash_app
def infer(
    script: str = None,
    input_data_dir: str = None,
    input_model_dir: str = None,
    output_dir: str = None,
    calc_infer_scores: bool = False,
    y_col_name: str = None,
    conda_env: str = None,
    stdout: str = "stdout.txt",
    stderr: str = "stderr.txt",
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = []):
    """Infer the model."""
    call = "echo 'Inferencing the model'"
    prefix = f"START=$(date +%s) ; echo Start:\t$START "

    if conda_env:
        conda= f"conda_path=$(dirname $(dirname $(which conda))) ; source $conda_path/bin/activate {conda_env} "
    else:
        conda = "echo no conda env provided"
    
    suffix = "STOP=$(date +%s) ; echo Duration:\t$((STOP-START)) seconds ; sleep 1"

   
    # Create the command line interface for inference
    cli = [ "time",
              script,
          "--input_data_dir" , input_data_dir,
          "--input_model_dir" , input_model_dir,
          "--output_dir" , output_dir,
          "--calc_infer_scores" , calc_infer_scores,
          "--y_col_name" , y_col_name
     ]

    # create a list of strings from cli
    call = " ;".join([prefix, conda, " ".join([ str(i) for i in cli]), suffix])

    logger.debug(f"Inference command: {call}")
    print(call)
    return call




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("parsl").setLevel(logging.INFO)

    print("Parsl version: ", parsl.__version__)

    parsl.load(config)

    print("Parsl config loaded")

    train().result()

    print("Training completed")

    with open('stdout.txt', 'r') as f:
        print(f.read())

    print("Done")
#     result = subprocess.run(infer_run, capture_output=True,

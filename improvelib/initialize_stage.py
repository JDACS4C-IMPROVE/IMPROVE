from improvelib.drp_param_def import *
from improvelib.benchmark_def import Benchmark
from improvelib.parsing_utils import finalize_parameters
import importlib

improve_preprocess_params = [
    {
        "name": "improvespecificpreprocessthing",
        "type": str,
        "default": "ABC",
        "help": "whatever.",
    },
]
improve_train_params = [
    {
        "name": "improvespecificTRAINthing",
        "type": str,
        "default": "ABC",
        "help": "whatever.",
    },
]
improve_infer_params = [
    {
        "name": "improvespecificINFERthing",
        "type": str,
        "default": "ABC",
        "help": "whatever.",
    },
]


def initialize_benchmark(filepath, defmodel, additional_definitions):
    bmk = Benchmark(
            filepath=filepath,
            defmodel=defmodel,
            framework="pytorch",
            prog="frm",
            desc="Framework functionality in IMPROVE",
            additional_definitions=additional_definitions,
            required=None,
        )
    params = finalize_parameters(bmk)
    return params
    

def determine_stage_specific_params(filename, filepath, application, defmodel, model_preprocess_params, model_train_params, model_infer_params):

    #model_param_def_fname = filepath + "/" + application + "_param_def"
    # get app specific params
    app_param_def_fname = "." + application + "_param_def"
    app_param_def = importlib.import_module(app_param_def_fname, 'improvelib') 
    #to do -- error handing
    if "preprocess" in filename:
        additional_definitions = improve_preprocess_params + app_preprocess_params + model_preprocess_params
        params = initialize_benchmark(filepath, defmodel, additional_definitions)
    elif "train" in filename:
        additional_definitions = improve_train_params + app_train_params + model_train_params
        params = initialize_benchmark(filepath, defmodel, additional_definitions)
    elif "infer" in filename:
        additional_definitions = improve_infer_params + app_infer_params + model_infer_params
        params = initialize_benchmark(filepath, defmodel, additional_definitions)
    else:
        print("File name does not contain 'preprocess', 'train', or 'infer'!")
    return params

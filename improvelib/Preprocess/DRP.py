# Library for drug response prediction specific functions for preprocessing


# Create new class for DrugResponsePrediction inheriting from Preprocess

from improvelib.preprocess import Preprocess


class DrugResponsePrediction(Preprocess):

    def __init__(self):
        super().__init__()

    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    drp = DrugResponsePrediction()
    model_params = [ { 'name' : "model_name"} , { 'name' : 'model_version'} ]
    drp.initialize_parameters( 
        pathToModelDir=None, 
        default_config=None,
        additional_definitions=model_params,
        required=["model_name", "model_version"] )

import os
from improve.config import Config

class Params:
    pass



class Preprocess(Config):

    # Set section for config file
    section = 'Preprocess'

    # Set options for command line
    preprocess_options = [
        {
            'name':'training_index_file',
            'default' : 'training.idx',
            'type': str,
            'help':'index file for training set [numpy array]'
        },
        {
            'name':'validation_index_file',
            'default' : 'validation.idx' ,
            'type': str,
            'help':'index file for validation set [numpy array]',
        },
        {
            'name':'testing_index_file',
            'default' : 'testing.idx' ,
            'type': str,
            'help':'index file for testiing set [numpy array]',
        },
        {
            'name':'data',
            'default' : 'data.parquet' ,
            'type': str,
            'help':'data file',
        },
        {
            'name':'input_type',
            'default' : 'BenchmarkV1' ,
            'choices' : ['parquet', 'csv', 'hdf5', 'npy', 'BenchmarkV1'],
            'metavar' : 'TYPE',
            'help' : 'Sets the input type. Default is BenchmarkV1. Other options are parquet, csv, hdf5, npy'
        }
    ]

    def __init__(self) -> None:
        super().__init__()
        self.options = Preprocess.preprocess_options



    def set_param(self, key, value):
        return super().set_param(Preprocess.section, key, value)


    def initialize_parameters(self, pathToModelDir, section='Preprocess', default_config='default.cfg', default_model=None, additional_definitions=None, required=None):
        if additional_definitions :
            self.options = self.options + additional_definitions
            print(self.options)
            sys.exit()

        return super().initialize_parameters(pathToModelDir, section, default_config, default_model, self.options , required)

    def add_params(self):
        pass
    

if __name__ == "__main__":
    p=Preprocess()
    p.initialize_parameters(pathToModelDir=".")

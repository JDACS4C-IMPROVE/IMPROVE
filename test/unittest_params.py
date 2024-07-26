import unittest
from unittest.mock import patch
import sys
from io import StringIO
#import logging

#logging.basicConfig(level=logging.DEBUG)

import test_preprocess_params
import test_train_params
import test_infer_params

config_file_1='test_default_1.cfg'
config_file_2='test_default_2.cfg'

class TestPreprocessConfigs(unittest.TestCase):
    @patch('sys.stdout')
    def test_preprocess_default_values(self, mock_stdout):
        """Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['test_preprocess_params.py']
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        self.assertTrue(preprocess_params['x_data_dir'] == 'x_data', msg="Parameter value is not the default value.")
        self.assertTrue(preprocess_params['y_data_dir'] == 'y_data', msg="Parameter value is not the default value.")
        
    @patch('sys.stdout')
    def test_preprocess_config_values(self, mock_stdout):
        #with patch('sys.stdout', new=StringIO()) as capture:
        #sys.argv = ['test_preprocess_params.py', '--splits_dir', 'test_splits']
        """Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['test_preprocess_params.py']
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        #output = mock_stdout.getvalue().strip()
        #output = capture.getvalue().strip()
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'preprocess', msg=f"Parameter value {preprocess_params['preprocess_test_var']} does not match the config file value in {config_file_1}.")
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits', msg=f"Parameter value {preprocess_params['splits_dir']} does not match the config file value in {config_file_1}.")
        
    @patch('sys.stdout')
    def test_preprocess_cli_values(self, mock_stdout):
        """Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['test_preprocess_params.py', '--splits_dir', 'test_splits_cli']
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        #output = mock_stdout.getvalue().strip()
        #output = capture.getvalue().strip()
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'preprocess', msg=f"Parameter value {preprocess_params['preprocess_test_var']} does not match the config file value in {config_file_1}.")
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_cli', msg=f"Parameter value {preprocess_params['splits_dir']} does not match the CLI value: test_splits_cli.")
    
    @patch('sys.stdout')    
    def test_preprocess_nondefault_config_values(self, mock_stdout):
        """Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['test_preprocess_params.py', '--config_file', config_file_2]
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'prep', msg="Parameter value is not the default value.") 
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_b', msg=f"Parameter value {preprocess_params['splits_dir']} does not match the config file value in {config_file_2}.")
        
#class TestTrainConfigs(unittest.TestCase):
    
#class TestInferConfigs(unittest.TestCase):   

if __name__ == "__main__":
    unittest.main(verbosity=2)

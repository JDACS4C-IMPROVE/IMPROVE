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

# TODO: Number tests? Save param_log_file.txt per test? Check global params?

class TestPreprocessConfigs(unittest.TestCase):
    
    @patch('sys.stdout')
    def test_preprocess_default_values(self, mock_stdout):
        """PREPROCESS: Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['test_preprocess_params.py']
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        self.assertTrue(preprocess_params['x_data_dir'] == 'x_data', msg="Parameter value is not the default value.")
        self.assertTrue(preprocess_params['y_data_dir'] == 'y_data', msg="Parameter value is not the default value.")
        
    @patch('sys.stdout')
    def test_preprocess_config_values(self, mock_stdout):
        #with patch('sys.stdout', new=StringIO()) as capture:
        #sys.argv = ['test_preprocess_params.py', '--splits_dir', 'test_splits']
        """PREPROCESS: Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['test_preprocess_params.py']
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        #output = mock_stdout.getvalue().strip()
        #output = capture.getvalue().strip()
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'preprocess', msg=f"Parameter value {preprocess_params['preprocess_test_var']} does not match the config file value in {config_file_1}.")
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits', msg=f"Parameter value {preprocess_params['splits_dir']} does not match the config file value in {config_file_1}.")
        
    @patch('sys.stdout')
    def test_preprocess_cli_values(self, mock_stdout):
        """PREPROCESS: Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['test_preprocess_params.py', '--splits_dir', 'test_splits_cli']
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        #output = mock_stdout.getvalue().strip()
        #output = capture.getvalue().strip()
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'preprocess', msg=f"Parameter value {preprocess_params['preprocess_test_var']} does not match the config file value in {config_file_1}.")
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_cli', msg=f"Parameter value {preprocess_params['splits_dir']} does not match the CLI value: test_splits_cli.")
    
    @patch('sys.stdout')    
    def test_preprocess_nondefault_config_values(self, mock_stdout):
        """PREPROCESS: Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['test_preprocess_params.py', '--config_file', config_file_2]
        preprocess_params = test_preprocess_params.main(sys.argv[1:])
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'prep', msg="Parameter value is not the default value.") 
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_b', msg=f"Parameter value {preprocess_params['splits_dir']} does not match the config file value in {config_file_2}.")
        
class TestTrainConfigs(unittest.TestCase):
    
    @patch('sys.stdout')
    def test_train_default_values(self, mock_stdout):
        """TRAIN: Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['test_train_params.py']
        train_params = test_train_params.main(sys.argv[1:])
        self.assertEqual(train_params['epochs'], 7, msg="Parameter value is not the default value.")
        self.assertEqual(train_params['learning_rate'], 7, msg="Parameter value is not the default value.")
        
    @patch('sys.stdout')
    def test_train_config_values(self, mock_stdout):
        """TRAIN: Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['test_train_params.py']
        train_params = test_train_params.main(sys.argv[1:])
        # TODO: Check why train_test_var is string value
        self.assertEqual(int(train_params['train_test_var']), 10, msg=f"Parameter value {train_params['train_test_var']} does not match the config file value in {config_file_1}.")
        self.assertTrue(train_params['model_file_name'] == 'test_model', msg=f"Parameter value {train_params['model_file_name']} does not match the config file value in {config_file_1}.")
        
    @patch('sys.stdout')
    def test_train_cli_values(self, mock_stdout):
        """TRAIN: Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['test_train_params.py', '--batch_size', '40']
        train_params = test_train_params.main(sys.argv[1:])
        # TODO: Check why train_test_var is string value
        self.assertEqual(int(train_params['train_test_var']), 10, msg=f"Parameter value {train_params['train_test_var']} does not match the config file value in {config_file_1}.")
        self.assertEqual(train_params['batch_size'], 40, msg=f"Parameter value {train_params['batch_size']} does not match the CLI value: test_splits_cli.")
    
    @patch('sys.stdout')    
    def test_train_nondefault_config_values(self, mock_stdout):
        """TRAIN: Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['test_train_params.py', '--config_file', config_file_2]
        train_params = test_train_params.main(sys.argv[1:])
        self.assertEqual(int(train_params['train_test_var']), 5, msg="Parameter value is not the default value.") 
        # TODO: Check why epochs is string value
        self.assertEqual(int(train_params['epochs']), 20, msg=f"Parameter value {train_params['epochs']} does not match the config file value in {config_file_2}.")
        
        
class TestInferConfigs(unittest.TestCase):   
    
    @patch('sys.stdout')
    def test_infer_default_values(self, mock_stdout):
        """INFER: Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['test_infer_params.py']
        infer_params = test_infer_params.main(sys.argv[1:])
        self.assertEqual(infer_params['test_batch'], 64, msg="Parameter value is not the default value.")
        self.assertTrue(infer_params['infer_test_var'] == "infer", msg="Parameter value is not the default value.")
        
    @patch('sys.stdout')
    def test_infer_config_values(self, mock_stdout):
        """INFER: Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['test_infer_params.py']
        infer_params = test_infer_params.main(sys.argv[1:])
        self.assertTrue(infer_params['json_scores_suffix'] == 'test_scores', msg=f"Parameter value {infer_params['json_scores_suffix']} does not match the config file value in {config_file_1}.")
        
    @patch('sys.stdout')
    def test_infer_cli_values(self, mock_stdout):
        """INFER: Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['test_infer_params.py', '--json_scores_suffix', 'test_scores_cli']
        infer_params = test_infer_params.main(sys.argv[1:])
        self.assertTrue(infer_params['json_scores_suffix'] == 'test_scores_cli', msg=f"Parameter value {infer_params['json_scores_suffix']} does not match the CLI value: test_splits_cli.")
    
    @patch('sys.stdout')    
    def test_infer_nondefault_config_values(self, mock_stdout):
        """INFER: Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['test_infer_params.py', '--config_file', config_file_2]
        infer_params = test_infer_params.main(sys.argv[1:])
        self.assertTrue(infer_params['infer_test_var'] == 'test_infer', msg=f"Parameter value {infer_params['infer_test_var']} does not match the config file value in {config_file_2}.")
        self.assertTrue(infer_params['json_scores_suffix'] == "scores", msg="Parameter value is not the default value.")
        
if __name__ == "__main__":
    unittest.main(verbosity=2)

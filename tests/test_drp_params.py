"""
Unit tests for checking the parameters are read and parsed correctly using two different config files (default and one set on the command line) with test drug response model (DRP) scripts for preprocess, train, and infer.
These include:
1. Tests to check parameters are read in this order: CLI, config file, default.
2. Tests to check types of the parameters match types listed in the parameter definitions.
3. Tests to check that values are read in correctly for parameters shared among different sections. 
"""
import unittest
import sys
import os
from pathlib import Path

# global variables
filepath = Path(__file__).resolve().parent

import model_preprocess_improve
import model_train_improve
import model_infer_improve

# set paths to test config files
config_file_1=str(filepath) + '/default.cfg'
config_file_2=str(filepath) + '/optional.cfg'

os.environ['IMPROVE_DEBUG_LEVEL'] = "WARNING"

class TestOrderPreprocessConfigs(unittest.TestCase):
    
    """
    Tests to check that the parameters are read in this order for PREPROCESS: CLI, config file, default.
    """ 
       
    def test_preprocess_default_values(self):
        """PREPROCESS: Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertTrue(preprocess_params['x_data_dir'] == 'x_data', msg="Parameter value is not the default value.")
        self.assertTrue(preprocess_params['y_data_dir'] == 'y_data', msg="Parameter value is not the default value.")
        
    def test_preprocess_config_values(self):
        """PREPROCESS: Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'prep', msg=f"Parameter value '{preprocess_params['preprocess_test_var']}' does not match the config file value in {config_file_1}.")
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits', msg=f"Parameter value '{preprocess_params['splits_dir']}' does not match the config file value in {config_file_1}.")
        
    def test_preprocess_cli_values(self):
        """PREPROCESS: Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['python model_preprocess_improve.py', '--splits_dir', 'test_splits_cli']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertTrue(preprocess_params['preprocess_test_var'] == 'prep', msg=f"Parameter value '{preprocess_params['preprocess_test_var']}' does not match the config file value in {config_file_1}.")
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_cli', msg=f"Parameter value '{preprocess_params['splits_dir']}' does not match the CLI value: test_splits_cli.")
        
    def test_preprocess_nondefault_config_values(self):
        """PREPROCESS: Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertTrue(preprocess_params['preprocess_test_var'] == '', msg="Parameter value is not the default value.") 
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_b', msg=f"Parameter value '{preprocess_params['splits_dir']}' does not match the config file value in {config_file_2}.")
        
    def test_preprocess_nondefault_config_and_cli_values(self):
        """PREPROCESS: Check CLI values of parameters when using non-default config file, config_file_2."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2, '--variable_name', 'cli_test']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertTrue(preprocess_params['preprocess_test_var'] == '', msg="Parameter value is not the default value.") 
        self.assertTrue(preprocess_params['splits_dir'] == 'test_splits_b', msg=f"Parameter value '{preprocess_params['splits_dir']}' does not match the config file value in {config_file_2}.")
        self.assertTrue(preprocess_params['variable_name'] == 'cli_test', msg=f"Parameter value '{preprocess_params['variable_name']}' does not match the CLI value: cli_test.")
        
class TestOrderTrainConfigs(unittest.TestCase):
    
    """
    Tests to check that the parameters are read in this order for TRAIN: CLI, config file, default.
    """ 
    
    def test_train_default_values(self):
        """TRAIN: Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['python train_params.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertEqual(train_params['epochs'], 7, msg="Parameter value is not the default value.")
        self.assertEqual(train_params['learning_rate'], 7, msg="Parameter value is not the default value.")
        
    def test_train_config_values(self):
        """TRAIN: Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['python train_params.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertEqual(train_params['train_test_var'], 10, msg=f"Parameter value '{train_params['train_test_var']}' does not match the config file value in {config_file_1}.")
        self.assertTrue(train_params['model_file_name'] == 'test_model', msg=f"Parameter value '{train_params['model_file_name']}' does not match the config file value in {config_file_1}.")
        
    def test_train_cli_values(self):
        """TRAIN: Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['python train_params.py', '--batch_size', '40']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertEqual(train_params['train_test_var'], 10, msg=f"Parameter value '{train_params['train_test_var']}' does not match the config file value in {config_file_1}.")
        self.assertEqual(train_params['batch_size'], 40, msg=f"Parameter value '{train_params['batch_size']}' does not match the CLI value: test_splits_cli.")
      
    def test_train_nondefault_config_values(self):
        """TRAIN: Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['python train_params.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertEqual(train_params['train_test_var'], 5, msg="Parameter value is not the default value.") 
        self.assertEqual(train_params['epochs'], 20, msg=f"Parameter value '{train_params['epochs']}' does not match the config file value in {config_file_2}.")
        
    def test_train_nondefault_config_and_cli_values(self):
        """TRAIN: Check CLI values of parameters when using non-default config file, config_file_2."""
        sys.argv = ['python train_params.py', '--config_file', config_file_2, '--variable_name', 'cli_test']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertEqual(train_params['train_test_var'], 5, msg="Parameter value is not the default value.") 
        self.assertEqual(train_params['epochs'], 20, msg=f"Parameter value '{train_params['epochs']}' does not match the config file value in {config_file_2}.")
        self.assertTrue(train_params['variable_name'] == 'cli_test', msg=f"Parameter value '{train_params['variable_name']}' does not match the CLI value: cli_test.")    

class TestOrderInferConfigs(unittest.TestCase):   
    
    """
    Tests to check that the parameters are read in this order for INFER: CLI, config file, default.
    """ 
    
    def test_infer_default_values(self):
        """INFER: Check default values of parameters when config_file_1 is set as default config file."""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertEqual(infer_params['test_batch'], 64, msg="Parameter value is not the default value.")
        self.assertTrue(infer_params['infer_test_var'] == "infer", msg="Parameter value is not the default value.")
        
    def test_infer_config_values(self):
        """INFER: Check config values of parameters listed in config_file_1 when this file is set as default config file."""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertTrue(infer_params['json_scores_suffix'] == 'test_scores', msg=f"Parameter value '{infer_params['json_scores_suffix']}' does not match the config file value in {config_file_1}.")
        
    def test_infer_cli_values(self):
        """INFER: Check CLI values of parameters with config_file_1 set as default config file."""
        sys.argv = ['python model_infer_improve.py', '--json_scores_suffix', 'test_scores_cli']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertTrue(infer_params['json_scores_suffix'] == 'test_scores_cli', msg=f"Parameter value '{infer_params['json_scores_suffix']}' does not match the CLI value: test_splits_cli.")
    
    def test_infer_nondefault_config_values(self):
        """INFER: Check config values of parameters listed in config_file_2 when this file is set using the CLI argument."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertTrue(infer_params['infer_test_var'] == 'test_infer', msg=f"Parameter value '{infer_params['infer_test_var']}' does not match the config file value in {config_file_2}.")
        self.assertTrue(infer_params['json_scores_suffix'] == "scores", msg="Parameter value is not the default value.")

    def test_infer_nondefault_config_and_cli_values(self):
        """INFER: Check CLI values of parameters when using non-default config file, config_file_2."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2, '--variable_name', 'cli_test']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertTrue(infer_params['infer_test_var'] == 'test_infer', msg=f"Parameter value '{infer_params['infer_test_var']}' does not match the config file value in {config_file_2}.")
        self.assertTrue(infer_params['variable_name'] == 'cli_test', msg=f"Parameter value {infer_params['variable_name']} does not match the CLI value: cli_test.")

class TestTypePreprocessConfigs(unittest.TestCase):
    
    """
    Tests to check the parameter types when read in the following formats for PREPROCESS: CLI, config file, default.
    """ 
    
    # Check types when parameters are set as default.
    
    def test_preprocess_default_type_string(self):
        """PREPROCESS: Check type for string parameter."""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['variable_name'], str)
        
    def test_preprocess_default_type_list(self):
        """PREPROCESS: Check type for list parameter"""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['split'], list)
        
    def test_preprocess_default_type_boolean(self):
        """PREPROCESS: Check type for boolean parameter."""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['only_cross_study'], bool)
        
    def test_preprocess_default_type_integer(self):
        """PREPROCESS: Check type for integer parameter."""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['study_number'], int)
        
    def test_preprocess_default_type_float(self):
        """PREPROCESS: Check type for float parameter."""
        sys.argv = ['python model_preprocess_improve.py']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['train_percent'], float)
        
    # Check types when parameters are set as in the config file.
  
    def test_preprocess_config_type_string(self):
        """PREPROCESS: Check type for string parameter in config_file_2."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['variable_name'], str)
             
    def test_preprocess_config_type_list(self):
        """PREPROCESS: Check type for list parameter in config_file_2."""        
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['split'], list)
        
    def test_preprocess_config_type_boolean(self):
        """PREPROCESS: Check type for boolean parameter in config_file_2."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['only_cross_study'], bool)
   
    def test_preprocess_config_type_integer(self):
        """PREPROCESS: Check type for integer parameter in config_file_2."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['study_number'], int)
        
    def test_preprocess_config_type_float(self):
        """PREPROCESS: Check type for float parameter in config_file_2."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['train_percent'], float)

    # Check types when parameters are set on the command line.
    
    def test_preprocess_cli_type_string(self):
        """PREPROCESS: Check type for string parameter when set on the command line."""
        sys.argv = ['python model_preprocess_improve.py', '--variable_name', 'cli_test']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['variable_name'], str)
             
    def test_preprocess_cli_type_list(self):
        """PREPROCESS: Check type for list parameter when set on the command line."""        
        sys.argv = ['python model_preprocess_improve.py', '--split', '4' ,'5']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['split'], list)
        
    def test_preprocess_cli_type_boolean(self):
        """PREPROCESS: Check type for boolean parameter when set on the command line."""
        sys.argv = ['python model_preprocess_improve.py', '--only_cross_study', 'True']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['only_cross_study'], bool)
   
    def test_preprocess_cli_type_integer(self):
        """PREPROCESS: Check type for integer parameter when set on the command line."""
        sys.argv = ['python model_preprocess_improve.py', '--study_number', '4']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['study_number'], int)
        
    def test_preprocess_cli_type_float(self):
        """PREPROCESS: Check type for float parameter when set on the command line."""
        sys.argv = ['python model_preprocess_improve.py', '--train_percent', '0.9']
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertIsInstance(preprocess_params['train_percent'], float) 
        
class TestTypeTrainConfigs(unittest.TestCase):
    
    """
    Tests to check the parameter types when read in the following formats for TRAIN: CLI, config file, default.
    """ 
    
    # Check types when parameters are set as default.
    
    def test_train_default_type_string(self):
        """TRAIN: Check type for string parameter."""
        sys.argv = ['python model_train_improve.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['variable_name'], str)
        
    def test_train_default_type_list(self):
        """TRAIN: Check type for list parameter"""
        sys.argv = ['python model_train_improve.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['split'], list)
        
    def test_train_default_type_boolean(self):
        """TRAIN: Check type for boolean parameter."""
        sys.argv = ['python model_train_improve.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['only_cross_study'], bool)
        
    def test_train_default_type_integer(self):
        """TRAIN: Check type for integer parameter."""
        sys.argv = ['python model_train_improve.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['study_number'], int)
        
    def test_train_default_type_float(self):
        """TRAIN: Check type for float parameter."""
        sys.argv = ['python model_train_improve.py']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['train_percent'], float)
        
    # Check types when parameters are set as in the config file.
  
    def test_train_config_type_string(self):
        """TRAIN: Check type for string parameter in config_file_2."""
        sys.argv = ['python model_train_improve.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['variable_name'], str)
             
    def test_train_config_type_list(self):
        """TRAIN: Check type for list parameter in config_file_2."""        
        sys.argv = ['python model_train_improve.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['split'], list)
        
    def test_train_config_type_boolean(self):
        """TRAIN: Check type for boolean parameter in config_file_2."""
        sys.argv = ['python model_train_improve.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['only_cross_study'], bool)
   
    def test_train_config_type_integer(self):
        """TRAIN: Check type for integer parameter in config_file_2."""
        sys.argv = ['python model_train_improve.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['study_number'], int)
        
    def test_train_config_type_float(self):
        """TRAIN: Check type for float parameter in config_file_2."""
        sys.argv = ['python model_train_improve.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['train_percent'], float)

    # Check types when parameters are set on the command line.
    
    def test_train_cli_type_string(self):
        """TRAIN: Check type for string parameter when set on the command line."""
        sys.argv = ['python model_train_improve.py', '--variable_name', 'cli_test']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['variable_name'], str)
             
    def test_train_cli_type_list(self):
        """TRAIN: Check type for list parameter when set on the command line."""        
        sys.argv = ['python model_train_improve.py', '--split', '4', '5' ]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['split'], list)
        
    def test_train_cli_type_boolean(self):
        """TRAIN: Check type for boolean parameter when set on the command line."""
        sys.argv = ['python model_train_improve.py', '--only_cross_study', 'True']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['only_cross_study'], bool)
   
    def test_train_cli_type_integer(self):
        """TRAIN: Check type for integer parameter when set on the command line."""
        sys.argv = ['python model_train_improve.py', '--study_number', '4']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['study_number'], int)
        
    def test_train_cli_type_float(self):
        """TRAIN: Check type for float parameter when set on the command line."""
        sys.argv = ['python train_params.py', '--train_percent', '0.9']
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertIsInstance(train_params['train_percent'], float)
        
class TestTypeInferConfigs(unittest.TestCase):
    
    """
    Tests to check the parameter types when read in the following formats for INFER: CLI, config file, default.
    """ 
    
    # Check types when parameters are set as default.
    
    def test_infer_default_type_string(self):
        """INFER: Check type for string parameter."""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['variable_name'], str)
        
    def test_infer_default_type_list(self):
        """INFER: Check type for list parameter"""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['split'], list)
        
    def test_infer_default_type_boolean(self):
        """INFER: Check type for boolean parameter."""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['only_cross_study'], bool)
        
    def test_infer_default_type_integer(self):
        """INFER: Check type for integer parameter."""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['study_number'], int)
        
    def test_infer_default_type_float(self):
        """INFER: Check type for float parameter."""
        sys.argv = ['python model_infer_improve.py']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['train_percent'], float)
        
    # Check types when parameters are set as in the config file.
  
    def test_infer_config_type_string(self):
        """INFER: Check type for string parameter in config_file_2."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['variable_name'], str)
             
    def test_infer_config_type_list(self):
        """INFER: Check type for list parameter in config_file_2."""        
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['split'], list)
        
    def test_infer_config_type_boolean(self):
        """INFER: Check type for boolean parameter in config_file_2."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['only_cross_study'], bool)
   
    def test_infer_config_type_integer(self):
        """INFER: Check type for integer parameter in config_file_2."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['study_number'], int)
        
    def test_infer_config_type_float(self):
        """INFER: Check type for float parameter in config_file_2."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['train_percent'], float)

    # Check types when parameters are set on the command line.
    
    def test_infer_cli_type_string(self):
        """INFER: Check type for string parameter when set on the command line."""
        sys.argv = ['python model_infer_improve.py', '--variable_name', 'cli_test']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['variable_name'], str)
             
    def test_infer_cli_type_list(self):
        """INFER: Check type for list parameter when set on the command line."""        
        sys.argv = ['python model_infer_improve.py', '--split', '[4,5]']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['split'], list)
        
    def test_infer_cli_type_boolean(self):
        """INFER: Check type for boolean parameter when set on the command line."""
        sys.argv = ['python model_infer_improve.py', '--only_cross_study', 'True']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['only_cross_study'], bool)
   
    def test_infer_cli_type_integer(self):
        """INFER: Check type for integer parameter when set on the command line."""
        sys.argv = ['python model_infer_improve.py', '--study_number', '4']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['study_number'], int)
        
    def test_infer_cli_type_float(self):
        """INFER: Check type for float parameter when set on the command line."""
        sys.argv = ['python model_infer_improve.py', '--train_percent', '0.9','--train_percent', '0.9']
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertIsInstance(infer_params['train_percent'], float)


class TestSeparationConfigs(unittest.TestCase):
    
    """
    Tests to check that sections that share the same parameter (ex. study_number) have the correct value.
    """ 

    def test_preprocess_same_variable(self):
        """PREPROCESS: Check type for string parameter in config_file_2."""
        sys.argv = ['python model_preprocess_improve.py', '--config_file', config_file_2]
        preprocess_params = model_preprocess_improve.main(sys.argv[1:])
        self.assertEqual(preprocess_params['study_number'], 2)   

    def test_train_same_variable(self):
        """TRAIN: Check type for string parameter in config_file_2."""
        sys.argv = ['python model_train_improve.py', '--config_file', config_file_2]
        train_params = model_train_improve.main(sys.argv[1:])
        self.assertEqual(train_params['study_number'], 3)   

    def test_infer_same_variable(self):
        """INFER: Check type for string parameter in config_file_2."""
        sys.argv = ['python model_infer_improve.py', '--config_file', config_file_2]
        infer_params = model_infer_improve.main(sys.argv[1:])
        self.assertEqual(infer_params['study_number'], 4)   
     
if __name__ == "__main__":
    unittest.main(verbosity=2)

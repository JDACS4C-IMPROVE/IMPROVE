import os
import logging
import unittest
import tempfile
import time
from pathlib import Path
from improvelib.initializer.config import Config


class TestConfig(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and config file for testing."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.test_dir.name) / "test.cfg"
        self.config_file.write_text(
            "[DEFAULT]\ninput_dir=./input\noutput_dir=./output\nlog_level=DEBUG\n")

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    def test_initialization(self):
        """Test the initialization of the Config class."""
        cfg = Config()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.log_level, os.getenv(
            "IMPROVE_LOG_LEVEL", logging.DEBUG))

    def test_load_config(self):
        """Test loading a configuration file."""
        cfg = Config()
        cfg.file = str(self.config_file)
        cfg.load_config()
        self.assertEqual(cfg.config['DEFAULT']['input_dir'], './input')
        self.assertEqual(cfg.config['DEFAULT']['output_dir'], './output')
        self.assertEqual(cfg.config['DEFAULT']['log_level'], 'DEBUG')

    def test_save_config(self):
        """Test saving a configuration file."""
        cfg = Config()
        cfg.set_param('DEFAULT', 'input_dir', './new_input')
        cfg.set_param('DEFAULT', 'output_dir', './new_output')
        cfg.save_config(str(self.config_file))
        cfg.file = self.config_file
        cfg.load_config()
        self.assertEqual(cfg.config['DEFAULT']['input_dir'], './new_input')
        self.assertEqual(cfg.config['DEFAULT']['output_dir'], './new_output')
        self.tearDown()
        self.setUp()

    def test_param(self):
        """Test getting and setting parameters."""
        cfg = Config()
        cfg.set_param('DEFAULT', 'test_key', 'test_value')
        value, error = cfg.param('DEFAULT', 'test_key')
        self.assertEqual(value, 'test_value')
        self.assertIsNone(error)

    def test_dict(self):
        """Test converting the config to a dictionary."""
        cfg = Config()
        cfg.file = str(self.config_file)
        cfg.load_config()
        config_dict = cfg.dict('DEFAULT')
        self.assertEqual(config_dict['input_dir'], './input')

    def test_load_cli_parameters(self):
        """Test loading CLI parameters from a file."""
        cli_params_file = Path(self.test_dir.name) / "cli_params.json"
        cli_params_file.write_text('{"param1": "value1", "param2": "value2"}')
        cfg = Config()
        params = cfg.load_cli_parameters(str(cli_params_file))
        self.assertEqual(params['param1'], 'value1')
        self.assertEqual(params['param2'], 'value2')


if __name__ == '__main__':
    unittest.main()

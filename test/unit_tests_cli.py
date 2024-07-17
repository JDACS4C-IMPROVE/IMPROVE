import unittest
from unittest.mock import patch
from improvelib.initializer.cli import CLI


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.cli = CLI()

    def test_add_parameters_with_same_name(self):
        options = [
            {'action': 'store', 'type': str, 'name': 'test_param'},
            {'action': 'store', 'type': str, 'name': 'test_param'}
        ]
        self.cli.set_command_line_options(options=options)
        self.assertEqual(
            len(self.cli.get_command_line_options()), 1, "Duplicate options should be removed")

    def test_add_same_parameter_to_different_groups(self):
        options = [
            {'action': 'store', 'type': str, 'name': 'test_param'}
        ]
        self.cli.set_command_line_options(options=options, group='test_group1')
        self.cli.set_command_line_options(options=options, group='test_group2')
        self.assertEqual(len(self.cli.get_command_line_options()), 1,
                         "Same parameter cannot be present in multiple groups")

    def test_predefined_options_not_overwritten(self):
        options = [
            {'action': 'store', 'type': str, 'name': 'input_dir'},
            {'action': 'store', 'type': str, 'name': 'output_dir'}
        ]
        self.cli.set_command_line_options(options=options)
        self.assertEqual(
            len(self.cli.get_command_line_options()), 0, "Predefined options should not be overwritten")


if __name__ == '__main__':
    unittest.main()

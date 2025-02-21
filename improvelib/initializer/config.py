"""Manages configuration settings and their processing order for IMPROVE model.

This module provides functionality to handle configuration files and command-line
options in a consistent sequence. It supports loading from INI/JSON/YAML files,
command-line argument parsing, and parameter validation.

Classes:
    Config: Handles configuration file loading and command-line option processing.
        Provides methods for parameter validation, type conversion, and
        environment setup.        
"""

import configparser
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional, List, Dict, Union, Any, Tuple

import yaml

from improvelib.initializer.cli import CLI
from improvelib.utils import str2bool


class Config:
    """Handles configuration files and command-line options.

    This class is responsible for managing configuration settings for the
    application. It provides methods to load, save, and update configuration
    parameters from files and command-line arguments.

    Attributes:
        config_sections (list of str): Sections of the configuration file.
        params (dict): Stores configuration parameters.
        file (str): Path to the configuration file.
        logger (logging.Logger): Logger for the class.
        log_level (int): Logging level.
        required (list of str): Required configuration parameters.
        config (configparser.ConfigParser): Parser for configuration files.
        cli (CLI): Command-line interface handler.
        input_dir (str): Input directory path.
        output_dir (str): Output directory path.
        _options (dict): Internal storage for command-line options.
    """

    config_sections = ['DEFAULT', 'Preprocess', 'Train', 'Infer']

    def __init__(self) -> None:
        """Initializes the Config class with default settings."""
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        required = ["input_dir", "output_dir", "log_level", 'config_file']

        self.params: Dict[str, Any] = {}
        self.file: Optional[str] = None  # change to config_file
        self.logger = logging.getLogger('Config')
        self.log_level = os.getenv("IMPROVE_LOG_LEVEL", logging.INFO)
        self.logger.setLevel(self.log_level)

        self.required = required
        self.config = configparser.ConfigParser()
        self.cli = CLI()
        # Default values are set in command line parser
        self.input_dir: Optional[str] = None
        self.output_dir: Optional[str] = None
        self._options: Dict[str, Dict[str, Any]] = {}

        # Set default directory paths based on environment variables.
        # Ensure that IMPROVE_DATA_DIR and CANDLE_DATA_DIR are identical if both are set.
        # Default IMPROVE_OUTPUT_DIR to IMPROVE_DATA_DIR or the current directory if not set.
        if "CANDLE_DATA_DIR" in os.environ and "IMPROVE_DATA_DIR" in os.environ:
            if os.getenv('IMPROVE_DATA_DIR') != os.getenv("CANDLE_DATA_DIR"):
                self.logger.error(
                    "Found CANDLE_DATA_DIR and IMPROVE_DATA_DIR but not identical."
                )
                raise ValueError('Alias not identical')
            else:
                self.config.set(
                    "DEFAULT", "input_dir", os.getenv("IMPROVE_DATA_DIR", "./")
                )

        elif "CANDLE_DATA_DIR" in os.environ:
            self.logger.debug("Setting IMPROVE_DATA_DIR to CANDLE_DATA_DIR")
            os.environ["IMPROVE_DATA_DIR"] = os.environ["CANDLE_DATA_DIR"]

        if "IMPROVE_OUTPUT_DIR" not in os.environ:
            self.logger.debug(
                'Setting IMPROVE_OUTPUT_DIR to IMPROVE_DATA_DIR or default'
            )
            os.environ["IMPROVE_OUTPUT_DIR"] = os.environ.get("IMPROVE_DATA_DIR", "./")

        self.config.set("DEFAULT", "input_dir", os.environ.get("IMPROVE_DATA_DIR", "./"))
        self.config.set("DEFAULT", "output_dir", os.environ.get("IMPROVE_OUTPUT_DIR", "./"))

    # ==========================================================
    # CONFIGURATION FILE METHODS
    # These methods handle the loading, saving, and management of configuration files.
    # They ensure that configuration data is correctly read from and written to files,
    # allowing the application to persist settings across sessions.
    # ==========================================================

    def load_config(self) -> None:
        """Loads configuration settings from specified file.

        Reads configuration from file into `config` attribute if file exists.
        Otherwise logs error and initializes empty DEFAULT section.

        Raises:
            FileNotFoundError: If configuration file does not exist or is inaccessible.
        """
        if self.file and os.path.isfile(self.file):
            self.logger.info("Loading config from %s", self.file)
            self.config.read(self.file)
        else:
            self.logger.error("Can't load config from %s", str(self.file))
            self.config['DEFAULT'] = {}

    def ini2dict(self, section: Optional[str] = None, flat: bool = False) -> Dict[str, Any]:
        """Converts INI configuration file to dictionary representation.

        Creates a dictionary from configuration options. If section is specified,
        returns only that section's options. If flat is True, combines all sections
        into a single dictionary without section keys.

        Args:
            section (Optional[str]): The section of the configuration to convert.
                If None, all sections are included.
            flat (bool): If True, returns a flat dictionary without sections.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration options.
        """
        params: Dict[str, Any] = {}

        # Determine which sections to process
        if section:
            # Check if the specified section exists
            if self.config.has_section(section):
                sections = [section]
            else:
                # Log an error if the section does not exist and return an empty dictionary
                self.logger.error("Can't find section %s", section)
                return params
        else:
            # If no specific section is provided, process all sections
            sections = self.config.sections()

        # Iterate over the determined sections
        for s in sections:
            if flat:
                # If flat is True, add all items to a single dictionary without section keys
                for key, value in self.config.items(s):
                    if key in params:
                        # Log a warning if a key collision is detected
                        self.logger.warning("Key collision detected for key: %s", key)
                    params[key] = value
            else:
                # Otherwise, organize items under their respective section keys
                params[s] = {key: value for key, value in self.config.items(s)}

        return params

    def dict(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Returns configuration as a dictionary (alias for ini2dict).

        Provides dictionary representation of configuration options. If section
        is specified, returns only options from that section.

        Args:
            section (Optional[str]): The section of the configuration to convert.
                If None, all sections are included.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration options for the
            specified section, or all sections if no section is specified.
        """
        return self.ini2dict(section=section)

    def save_parameter_file(self, file_name: Optional[str]) -> None:
        """Saves current parameters to a specified file.

        Writes parameters to the given file path. If path is relative, saves to
        `output_dir`. Creates directory if it doesn't exist.

        Args:
            file_name (Optional[str]): The name of the file to save the
                parameters to.

        Raises:
            ValueError: If `self.output_dir` is invalid.
            IOError: If there is an error writing the parameters to the file.
        """
        if not file_name:
            file_name = "config.yaml"
            self.logger.info("No file name provided. Using default: %s", file_name)

        if not hasattr(self, "output_dir") or not self.output_dir:
            raise ValueError("output_dir is not set or is invalid.")

        # Ensure `self.output_dir` is a Path
        output_dir = Path(self.output_dir)

        # Construct the full path
        if Path(file_name).is_absolute():
            path = Path(file_name)
        else:
            # Construct the path in the output directory
            path = output_dir / file_name
            # Create the directory if it does not exist
            if not Path(path.parent).exists():
                self.logger.debug(
                    "Creating directory %s for saving config file.", path.parent
                )
                Path(path.parent).mkdir(parents=True, exist_ok=True)

        try:
            # Write the parameters to the file
            with path.open("w") as f:
                f.write(str(self.params))
        except IOError as e:
            self.logger.error("Failed to save parameters to %s: %s", path, e)

    # ==========================================================
    # COMMAND LINE INTERFACE METHODS
    # These methods manage the parsing and handling of command line arguments.
    # They set up command line options, retrieve user inputs, and update defaults,
    # enabling dynamic configuration of the application via the command line.
    # ==========================================================
    def set_command_line_options(
        self, 
        options: Optional[List[Dict[str, Any]]] = None,
        group: Optional[str] = None
    ) -> bool:
        """Sets up command line options and updates internal tracking.

        Configures CLI options through CLI class and updates internal _options
        dictionary to track all command-line settings.

        Args:
            options (Optional[List[Dict[str, Any]]]): Dictionaries defining command
                line options.
            group (Optional[str]): Name of argument group to add options to.

        Returns:
            bool: True if command line options were successfully set.
            
        Raises:
            Exception: If setting command line options fails.
        """
        if options is None:
            options = []

        try:
            self.cli.set_command_line_options(options)
            self._update_options()
            return True
        except Exception as e:
            self.logger.error("Failed to set command line options: %s", e)
            return False

    def get_command_line_options(self) -> Dict[str, Any]:
        """Retrieves parsed command line options after updating defaults.

        Updates command line defaults from current configuration and returns
        the parsed command line arguments from the CLI class.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed command line options.
            
        Raises:
            Exception: If retrieving command line options fails.
        """
        try:
            self._update_cli_defaults()
            return self.cli.get_command_line_options()
        except Exception as e:
            self.logger.error("Failed to retrieve command line options: %s", e)
            return {}

    def load_cli_parameters(
        self, 
        file: str, 
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """Loads and validates command line parameter definitions from JSON/YAML file.

        Reads parameter definitions from file and validates them against expected
        criteria. Supports JSON and YAML formats.

        Args:
            file (str): The path to the file containing parameter definitions.
            section (Optional[str]): Reserved for future use to load specific
                sections, if applicable.

        Returns:
            Dict[str, Any]: Loaded parameters, where each key is a parameter name
                and value is its configuration.

        Raises:
            FileNotFoundError: If the file cannot be found.
            ValueError: If the file is in an unsupported format.
        """
        # Log the start of the parameter loading process
        self.logger.debug("Loading parameters from %s", file)

        # Convert Path to string if necessary for compatibility
        if file and isinstance(file, Path):
            file = str(file)

        # Check if the file exists
        if os.path.isfile(file):
            params = None  # Initialize params to None

            # Load parameters based on file extension
            if file.endswith('.json'):
                # Load JSON file
                with open(file, 'r') as f:
                    params = json.load(f)
            elif file.endswith('.yaml') or file.endswith('.yml'):
                # Load YAML file
                with open(file, 'r') as f:
                    params = yaml.safe_load(f)
            else:
                # Log an error and raise an exception for unsupported formats
                self.logger.error("Unsupported file format")
                raise ValueError("Unsupported file format")

            # Validate the loaded parameters to ensure they meet expected criteria
            self._validate_parameters(params)
            return params
        else:
            # Log a critical error and raise an exception if the file is not found
            self.logger.critical("Can't find file %s", file)
            raise FileNotFoundError(f"Can't find file {file}")

    def update_defaults(
        self,
        cli_definitions: Optional[List[Dict[str, Any]]] = None,
        new_defaults: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Updates default values for command line arguments with new values.

        Modifies command line definitions with new default values and updates
        parser defaults for existing options. Handles type conversion for various
        data types including lists, bools, ints, strings, and floats.

        Args:
            cli_definitions (Optional[List[Dict[str, Any]]]): List of dictionaries 
                defining command line options.
            new_defaults (Optional[Dict[str, Any]]): Dictionary containing new default 
                values for options.

        Returns:
            List[Dict[str, Any]]: Updated command line definitions with new defaults.

        Raises:
            json.JSONDecodeError: If a default value cannot be converted to a list.
        """
        # Get the list of added options from the parser
        existing_options = [o.lstrip('-') for o in self.cli.parser._option_string_actions]

        if not new_defaults:
            self.logger.error("No new defaults provided.")
            return []
        if not cli_definitions:
            self.logger.error("No command line definitions provided.")
            return []

        # Initialize the target dictionary
        updated_parameters = []

        # Loop through the command line definitions and update the default values
        # if the name is in the new defaults
        for entry in cli_definitions:
            self.logger.debug("Updating " + str(entry))
            if entry['name'] in new_defaults:
                entry['default'] = new_defaults[entry['name']]
                # Convert the default value to the correct type
                # The presence of nargs indicates that the default value is a list
                if "nargs" in entry:
                    try:
                        entry['default'] = json.loads(new_defaults[entry['name']])
                    except json.JSONDecodeError:
                        self.logger.error("Can't convert %s to list", new_defaults[entry['name']])
                        self.logger.error(json.JSONDecodeError)
                elif "type" in entry:
                    if entry['type'] == bool:
                        entry['default'] = str2bool(entry['default'])
                    elif entry['type'] == int:
                        entry['default'] = int(entry['default'])
                    elif entry['type'] == str:
                        entry['default'] = str(entry['default'])
                    elif entry['type'] == float:
                        entry['default'] = float(entry['default'])
                else:
                    self.logger.error("No type provided for " + str(entry['name']))

                # Update the default value in the parser if the option is already there
                if entry['name'] in existing_options:
                    self.cli.parser.set_defaults(**{entry['name']: entry['default']})

            # Append the updated entry to the list
            updated_parameters.append(entry)

        return updated_parameters

    def update_cli_definitions(
        self,
        definitions: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Updates command line definitions with values from config file.

        Extracts config file name from CLI args, loads the file, and updates the
        provided command line definitions. Should be called before
        `self.cli.set_command_line_options(options=updated_parameters)`.

        Args:
            definitions (Optional[List[Dict[str, Any]]]): List of dictionaries defining
                command line options to update with config file values.

        Returns:
            List[Dict[str, Any]]: Updated command line definitions with values from
                config file.
        """
        # Get the config file from the command line arguments otherwise use the default from self.file
        config_file_from_cli = self.cli.get_config_file()

        # Set self.file
        if config_file_from_cli is not None:
            self.file = config_file_from_cli
        else:
            self.logger.debug("No config file provided in command line arguments.")

        if self.file is None:
            self.logger.debug("No config file provided at all.")
            return []

        # Load the config file
        self.load_config()

        # Update additional_definitions with values from config file
        return self.update_defaults(cli_definitions=definitions, new_defaults=self.ini2dict(flat=True))

    def _add_option(self, name: str, option: Dict[str, Any]) -> bool:
        """Adds and validates a command line option to internal _options dictionary.

        Validates option definition by checking required keys, name matching, and
        supported types. Adds valid options to _options dictionary.

        Args:
            name (str): Name of the command line option.
            option (Dict[str, Any]): Dictionary defining the command line option,
                containing 'name', 'type', 'default', and 'help' keys.

        Returns:
            bool: True if option was successfully added, False if already defined
                or invalid.

        Raises:
            SystemExit: If option name doesn't match dictionary name.
        """
        # Check if option is a dictionary
        if not isinstance(option, dict):
            self.logger.error("Option %s is not a dictionary", name)
            return False

        # Check if name is identical to the name in the dictionary
        if "name" in option:
            if not name == option['name']:
                self.logger.error(
                    "Option name %s is not identical to name in dictionary %s",
                    name,
                    option['name']
                )
                sys.exit(1)
        elif not name == option['dest']:
            self.logger.error(
                "Option name %s is not identical to name in dictionary %s",
                name,
                option['dest']
            )
            return False

        # Check if name is already in _options
        if name in self._options:
            self.logger.error("Option %s is already defined. Skipping.", name)
            return False

        # Check if all required keys are present
        if not all(k in option for k in ('name', 'type', 'default', 'help')):
            self.logger.warning("Option %s is missing required keys.", name)

        # Check if type and default are supported
        if "type" not in option:
            self.logger.error("Option %s is missing type. Setting to str.", name)
            option['type'] = str
        if "default" not in option:
            self.logger.error("Option %s is missing default. Setting to None.", name)
            option['default'] = None

        # Use a set for supported types
        supported_types = {
            str, int, float, bool, str2bool,
            'str', 'int', 'float', 'bool', 'str2bool', None
        }
        if option['type'] not in supported_types:
            self.logger.error("Unsupported type %s for option %s", option['type'], name)
            return False

        # Add option to _options
        self._options[name] = option
        return True

    def _update_options(self) -> bool:
        """Updates internal _options dictionary with CLI arguments.

        Updates internal options from current CLI arguments. Should be called
        after adding new command line options via set_command_line_options.

        Returns:
            bool: True if all options were successfully updated, False if any
                option failed to be added.

        Raises:
            SystemExit: If option name doesn't match dictionary name in _add_option.
        """
        self.logger.debug("Starting to update internal options with command line arguments.")

        # Iterate over all actions in the CLI parser
        for action in self.cli.parser._actions:
            # Add each action's destination and attributes to the internal options
            if not self._add_option(action.dest, action.__dict__):
                self.logger.error("Failed to add option: %s", action.dest)
                return False

        self.logger.debug("Successfully updated all internal options.")
        return True

    def _update_cli_defaults(self) -> bool:
        """Updates CLI parser defaults from configuration file values.

        Reads configuration file (from CLI args or default) and updates CLI parser
        defaults for each matching option, with type conversion as needed.

        Returns:
            bool: True if defaults were successfully updated, False if no config
                file available.

        Raises:
            ValueError: If value cannot be converted to required type.
            TypeError: If value is of incompatible type for conversion.
            json.JSONDecodeError: If list value cannot be parsed from JSON.
        """
        # Retrieve the config file from command line arguments
        config_file = self.cli.get_config_file()

        if config_file is not None:
            self.file = config_file
        else:
            self.logger.debug("No config file provided in command line arguments.")

        # If no config file is available, log a message and exit the function
        if self.file is None:
            self.logger.debug("No config file provided at all.")
            return False

        # Load the configuration from the specified file
        self.load_config()

        # Iterate over each section in the configuration file
        for section in self.config.sections():
            # Check if the current section is the one we are interested in, or if no specific section is set
            if self.section is None or self.section == section:
                # Iterate over each option in the current section
                for key, raw_value in self.config.items(section):
                    value: Union[str, int, float, bool, list[Any]] = raw_value
                    if key in self._options:
                        # If the option expects a list, attempt to parse the value as JSON
                        if 'nargs' in self._options[key] and self._options[key]['nargs']:
                            if isinstance(raw_value, str):
                                try:
                                    value = json.loads(raw_value)
                                except json.JSONDecodeError:
                                    self.logger.error("Can't convert %s to list", raw_value)
                                    continue
                            else:
                                self.logger.error("Value for key %s must be a string to parse as JSON.", key)
                                continue

                        # If a specific type is set for the option, convert the value to that type
                        elif 'type' in self._options[key]:
                            t = self._options[key]['type']
                            try:
                                if t == 'str' or t == str:
                                    value = str(raw_value)
                                elif t == 'int' or t == int:
                                    if isinstance(raw_value, (str, int)):
                                        value = int(raw_value)
                                    else:
                                        raise ValueError(f"Cannot convert {raw_value} to int.")
                                elif t == 'float' or t == float:
                                    if isinstance(raw_value, (str, int, float)):
                                        value = float(raw_value)
                                    else:
                                        raise ValueError(f"Cannot convert {raw_value} to float.")
                                elif t == 'bool' or t == bool:
                                    if isinstance(raw_value, str):
                                        value = str2bool(raw_value)
                                    else:
                                        raise ValueError(f"Cannot convert {raw_value} to bool.")
                                elif t == 'str2bool':
                                    if isinstance(raw_value, str):
                                        value = str2bool(raw_value)
                                    else:
                                        raise ValueError(f"Cannot convert {raw_value} to str2bool.")
                                else:
                                    self.logger.error("Unsupported type %s for key %s", t, key)
                                    value = str(raw_value)
                            except (ValueError, TypeError) as e:
                                self.logger.error("Failed to convert %s to type %s: %s", raw_value, t, e)
                                continue
                        
                        # Update the default value for the option in the CLI parser
                        self.cli.parser.set_defaults(**{key: value})

        # Return True to indicate that the defaults were successfully updated
        return True

    # ==========================================================
    # PARAMETER MANAGEMENT METHODS
    # These methods handle the retrieval, setting, and validation of configuration parameters.
    # They provide functionality to access and modify parameter values, ensuring that
    # the application's configuration is consistent and meets required criteria.
    # ==========================================================

    def get_param(self, section: str = "DEFAULT", key: Optional[str] = None) -> str:
        """Retrieves configuration value from specified section.

        Gets value from config section (DEFAULT if not specified). Valid sections
        are: 'Preprocess', 'Train', and 'Infer'.

        Args:
            section (str): Section to access, defaults to 'DEFAULT'.
            key (Optional[str]): Configuration option key to retrieve.

        Returns:
            str: Value of the configuration option.

        Raises:
            ValueError: If key is None or option not found in section.
        """
        # Validate key is provided
        if key is None:
            raise ValueError("Key must be provided.")

        # Attempt to retrieve the value for the specified key in the section
        if self.config.has_option(section, key):
            value = self.config[section][key]
        else:
            # Log an error and raise an exception if the key does not exist
            error = f"Can't find option: {key}"
            self.logger.error(error)
            raise ValueError(error)

        return value

    def set_param(
        self, 
        section: str = "DEFAULT", 
        key: Optional[str] = None, 
        value: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """Sets configuration value in specified section.

        Sets value in config section (DEFAULT if not specified). Valid sections
        are: 'Preprocess', 'Train', and 'Infer'. Creates section if needed.

        Args:
            section (str): Section to access, defaults to 'DEFAULT'.
            key (Optional[str]): Configuration option key to set.
            value (Optional[str]): Value to set for the key. Empty string if None.

        Returns:
            Tuple[str, Optional[str]]: Tuple of (set value, error message or None).

        Raises:
            ValueError: If key is None.
        """
        # Initialize message as None
        msg = None

        # Validate key is provided
        if key is None:
            msg = "Can't update config, empty key"
            self.logger.error(msg)
            raise ValueError(msg)

        # Check if the section exists, create it if it doesn't and it's not "DEFAULT"
        if not self.config.has_section(section) and section != "DEFAULT":
            msg = "Unknown section " + str(section)
            self.logger.debug(msg)
            self.config.add_section(section)

        # Set the value, defaulting to an empty string if None
        if value is None:
            value = ''

        # Log the key and value being set
        self.logger.debug("Key:%s\tValue:%s", key, value)
        self.config[section][key] = str(value)

        # Return the set value and any message
        return (self.config[section][key], msg)

    def _validate_parameters(
        self, 
        params: Optional[List[Dict[str, Any]]], 
        required: Optional[List[str]] = None
    ) -> None:
        """Validates parameter types and checks for required parameters.

        Validates each parameter's type, converting string type names to Python types.
        Checks for presence of required parameters if specified.

        Args:
            params (Optional[List[Dict[str, Any]]]): Parameter dictionaries to validate,
                each containing a 'type' key.
            required (Optional[List[str]]): Required parameter names to check for
                in params list.

        Raises:
            ValueError: If required parameter missing or type unsupported.
        """
        if params is None:
            return

        # Iterate over each parameter dictionary in the list
        for p in params:
            # Check if 'type' is specified and convert to the corresponding Python type
            if 'type' in p:
                if p['type'] == 'str':
                    p['type'] = str
                elif p['type'] == 'int':
                    p['type'] = int
                elif p['type'] == 'float':
                    p['type'] = float
                elif p['type'] == 'bool':
                    p['type'] = bool
                elif p['type'] == 'str2bool':
                    p['type'] = str2bool
                else:
                    # Log an error and raise an exception for unsupported types
                    self.logger.error("Unsupported type %s", p['type'])
                    raise ValueError(f"Unsupported type: {p['type']}")

            # Check for required parameters if the 'required' list is provided
            if required:
                for req in required:
                    if req not in [param.get('name') for param in params]:
                        self.logger.error("Missing required parameter: %s", req)
                        raise ValueError(f"Missing required parameter: {req}")

    def load_parameter_definitions(
        self, 
        file: Union[str, Path]
    ) -> Optional[List[Dict[str, Any]]]:
        """Loads and validates parameter definitions from JSON/YAML file.

        Reads parameter definitions from file, validates them against expected
        criteria, and returns them as a list of dictionaries.

        Args:
            file (Union[str, Path]): Path to file containing parameter definitions.

        Returns:
            Optional[List[Dict[str, Any]]]: Parameter dictionaries if file loaded
                successfully, None otherwise.

        Raises:
            FileNotFoundError: If file cannot be found.
            ValueError: If file format unsupported or data corrupted.
            json.JSONDecodeError: If JSON parsing fails.
            yaml.YAMLError: If YAML parsing fails.
        """
        # Log the start of the parameter loading process
        self.logger.debug("Loading parameters from %s", file)

        # Ensure `file` is a string
        if isinstance(file, Path):
            file = str(file)

        # Check if the file exists
        if not os.path.isfile(file):
            self.logger.critical("Can't find file %s", file)
            raise FileNotFoundError(f"Can't find file: {file}")

        # Determine the file format and load accordingly
        try:
            if file.endswith('.json'):
                # Load JSON file
                with open(file, 'r') as f:
                    params = json.load(f)
            elif file.endswith('.yaml') or file.endswith('.yml'):
                # Load YAML file
                with open(file, 'r') as f:
                    params = yaml.safe_load(f)
            else:
                self.logger.error("Unsupported file format: %s", file)
                raise ValueError(f"Unsupported file format: {file}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error("Failed to parse the file %s: %s", file, e)
            raise ValueError(f"Invalid file format or corrupted data in {file}: {e}")

        # Validate the loaded parameters
        if params is not None:
            self._validate_parameters(params)
            return params

        # If params is still None, log and return None explicitly
        self.logger.warning("No parameters were loaded from file: %s", file)
        return None

    def section_parameters(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Returns configuration options from specified section or all sections.

        Retrieves configuration options from config file. If section is specified,
        returns only that section's options, otherwise returns all sections.

        Args:
            section (Optional[str]): Section to retrieve options from. If None,
                includes all sections.

        Returns:
            Dict[str, Any]: Configuration options from specified section(s).

        Raises:
            ValueError: If specified section does not exist.
        """
        # Initialize an empty dictionary to store parameters
        params = {}

        # Determine which sections to process
        sections = [section] if section else self.config.sections()

        # Iterate over the determined sections
        for s in sections:
            # Check if the specified section exists
            if not self.config.has_section(s):
                self.logger.error("Can't find section %s", s)
                raise ValueError(f"Can't find section: {s}")

            # Add items from the section to the params dictionary
            params[s] = {key: value for key, value in self.config.items(s)}

        return params

    def initialize_parameters(
        self,
        pathToModelDir: Union[str, Path],
        section: str = 'DEFAULT',
        default_config: Optional[Union[str, Path]] = None,
        additional_definitions: Optional[List[Dict[str, Any]]] = None,
        required: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Initializes configuration from file and command line arguments.

        Loads configuration defaults from file, updates with CLI arguments,
        sets environment variables, and ensures required directories exist.

        Args:
            pathToModelDir (Union[str, Path]): Path to model directory.
            section (str): Configuration section to initialize, defaults to 'DEFAULT'.
            default_config (Optional[Union[str, Path]]): Path to default config file.
            additional_definitions (Optional[List[Dict[str, Any]]]): Additional CLI
                argument definitions.
            required (Optional[List[str]]): Required parameter names to check.

        Returns:
            Dict[str, Any]: Dictionary of initialized parameters.

        Raises:
            FileNotFoundError: If default config file not found.
            RuntimeError: If required definitions or directories missing.
            SystemExit: If no additional definitions provided or config file not found.
        """
        # Set the logger level
        self.logger.setLevel(self.log_level)
        self.logger.debug("Initializing parameters for %s", section)

        # Preserve the type of the object
        current_class = self.__class__
        self.__class__ = Config

        # Set the section for reading the config file
        self.section = section

        # Check if the default config file is provided and reachable
        if default_config:
            if pathToModelDir:
                # Convert pathToModelDir to Path if necessary
                if not isinstance(pathToModelDir, Path):
                    pathToModelDir = Path(pathToModelDir)

                # Construct the full path to the default config file
                if not default_config.startswith("/"):
                    default_config = pathToModelDir / default_config
                else:
                    self.logger.error("No path to model directory provided.")
            if not os.path.isfile(default_config):
                self.logger.error("Can't find default config file %s", default_config)
                sys.exit(1)
            else:
                self.logger.debug("Default config file found: %s", default_config)
                self.file = str(default_config)
        else:
            self.logger.warning("No default config file provided.")

        # Set and get command line arguments
        if additional_definitions:
            self.logger.debug("Updating additional definitions with values from config file.")
            # updated_definitions = self.update_cli_definitions(definitions=additional_definitions)
        else:
            self.logger.debug("No additional definitions provided.")
            sys.exit(0)

        # Set command line options
        self.set_command_line_options(options=additional_definitions)
        if not self.set_command_line_options(options=additional_definitions):
            raise RuntimeError("Failed to set command-line options.")
        # Get command line options
        self.params = self.get_command_line_options()

        # Ensure CLI args exist and assign attributes
        # Set input and output directories
        if not hasattr(self.cli, 'args') or self.cli.args is None:
            self.logger.warning("CLI arguments not initialized. Using default values.")
            self.input_dir = "./"
            self.output_dir = "./"
            self.log_level = "INFO"
        else:
            self.input_dir = getattr(self.cli.args, 'input_dir', "./")
            self.output_dir = getattr(self.cli.args, 'output_dir', "./")
            self.log_level = getattr(self.cli.args, 'log_level', "INFO")
            self.logger.setLevel(self.log_level)
        
        self.logger.debug("Current log level is %s", self.log_level)

        # Update log level if set by command line
        if "log_level" in self.cli.params:
            self.logger.info("Log level set by command line, updating to %s", self.cli.params["log_level"])
            self.log_level = self.params["log_level"]
            self.logger.setLevel(self.log_level)
            
        # Ensure log_level is a string and handle None
        self.log_level = str(self.log_level) if self.log_level is not None else "INFO"

        self.logger.debug("Final parameters: %s", self.cli.cli_params)
        self.logger.debug("Final parameters: %s", self.params)
        self.logger.debug("Final parameters set.")

        # Set supported environment variables
        os.environ["IMPROVE_DATA_DIR"] = self.input_dir
        os.environ["IMPROVE_OUTPUT_DIR"] = self.output_dir
        os.environ["IMPROVE_LOG_LEVEL"] = self.log_level

        # Create output directory if it does not exist
        if not os.path.isdir(self.output_dir):
            self.logger.debug("Creating output directory: %s", self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

        # Save parameters to file if specified
        if "param_log_file" in self.params:
            final_config_file = self.params["param_log_file"]
            self.save_parameter_file(final_config_file)

        # Restore the original class type
        self.__class__ = current_class
        return self.params


if __name__ == "__main__":
    # ==========================================================
    # TEST/DEBUGGING BLOCK
    # This section is used for testing and debugging the Config class
    # functionality. It demonstrates how to initialize the Config class, load
    # parameters, and interact with command line options.
    # ==========================================================

    # Initialize the Config class
    cfg = Config()

    # Define common parameters for testing
    common_parameters = [
        {
            "name": "list_of_int",
            "dest": "loint",
            "help": "Need help to display default value",
            "nargs": "+",
            "type": int,
            "default": [100],
            "section": "DEFAULT"
        },
        {
            "name": "list_of_strings",
            "dest": "lostr",
            "nargs": "+",
            "type": str,
            "default": ['100'],
            "section": "DEFAULT"
        },
        {
            "name": "list_of_lists",
            "nargs": "+",
            "metavar": "lol",
            "dest": "l",
            "action": "append",
            "type": str,
            "default": [[1, 2, 3], [4, 5, 6]],
            "section": "DEFAULT"
        },
    ]

    # Define directories for loading additional parameters and configuration files
    current_dir = Path(__file__).resolve().parent
    test_dir = current_dir.parents[1] / "tests"

    # Load additional command line parameters from a file
    param_file = test_dir / "data/additional_command_line_parameters.yml"
    params = cfg.load_cli_parameters(str(param_file))  
    print("Loaded CLI Parameters:", params)

    # Set up argparse for testing command line options
    import argparse
    cfg_parser = argparse.ArgumentParser(
        description='Get the config file from command line.', add_help=False
    )
    cfg_parser.add_argument('--config_file', metavar='INI_FILE', type=str, dest="config_file")

    # Simulate command line arguments for testing
    sys.argv.extend(["--config_file", str(test_dir / "data/default.cfg")])

    cfg.cli.parser.add_argument(
        '--test', metavar='TEST_COMMAND_LINE_OPTION', dest="test",
        nargs='+', type=int, default=[1], help="Test command line option."
    )

    # Initialize parameters
    try:
        final_params = cfg.initialize_parameters(
            pathToModelDir="./",
            additional_definitions=common_parameters + params
        )
        print("Initialized Parameters:", final_params)
    except Exception as e:
        print(f"Error during parameter initialization: {e}")

    # Output results
    print("Initialized Parameters:", final_params)

    # Output the results to verify correct processing and storage of parameters
    print("Config Items in 'DEFAULT':", cfg.config.items('DEFAULT', raw=False))
    print("Parsed CLI Arguments:", cfg.cli.args)
    print("Final Parameters:", cfg.params)
"""
This module is responsible for managing the order of configuration settings within the IMPROVE model.
It ensures that configuration parameters are processed in a logical and consistent sequence, which is
crucial for maintaining the integrity and functionality of the application.

Classes:
    Config: A class to handle configuration files and command-line options.

Functions:
    __init__: Initializes the Config class with default settings.
    
    load_config: Loads the configuration from a file. [used]
    load_config_file: Loads the configuration file, setting paths and defaults. [not used]
    ini2dict: Converts INI configuration to a dictionary. [used]
    dict: Alias for ini2dict, returns configuration as a dictionary. [used]
    save_parameter_file: Saves final parameters to a file. [used]
    save_config: Saves the configuration to a file. [removed]
    
    set_command_line_options: Sets up command line options. [used]
    get_command_line_options: Retrieves parsed command line arguments. [keep]
    load_cli_parameters: Loads command line parameter definitions from a file. [keep]
    update_defaults: Updates default values for command line arguments. [used]
    update_cli_definitions: Updates CLI argument definitions with values from the config file. [keep]
    _add_option: Adds a command line option definition to _options. [used]
    _update_options: Updates _options with options from the command line. [used]
    _update_cli_defaults: Updates command line defaults with values from _options. [used]
    
    param: Gets or sets a value for a given option. [removed]
    get_param: Gets a value for a given option. [used]
    set_param: Sets a value for a given option. [keep]
    check_required: Checks if all required parameters are set. (Empty) [removed]
    _validate_parameters: Validates parameters, setting types and checking for required parameters. [used]
    load_parameter_definitions: Loads parameter definitions from a file.
    validate_parameters: Validates parameters. (Removed)
    section_parameters: Returns a dictionary of all options in a section.
    initialize_parameters: Initializes parameters from the command line and config file. (Check for overlap)
"""
import configparser
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional, List, Dict

import yaml

from improvelib.initializer.cli import CLI
from improvelib.utils import cast_value, str2bool




class Config:
    """Handles configuration files and command-line options.

    This class is responsible for managing configuration settings for the application.
    It provides methods to load, save, and update configuration parameters from files
    and command-line arguments.

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
        # Default format
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        required = ["input_dir", "output_dir", "log_level", 'config_file']

        self.params = {}
        self.file = None  # change to config_file
        self.logger = logging.getLogger('Config')
        self.log_level = os.getenv("IMPROVE_LOG_LEVEL", logging.INFO)
        self.logger.setLevel(self.log_level)

        self.required = required
        self.config = configparser.ConfigParser()
        self.cli = CLI()
        # Default values are set in command line parser
        self.input_dir = None
        self.output_dir = None
        self._options = {}

        # Set default directory paths based on environment variables.
        # Ensure that IMPROVE_DATA_DIR and CANDLE_DATA_DIR are identical if both are set.
        # Default IMPROVE_OUTPUT_DIR to IMPROVE_DATA_DIR or the current directory if not set.
        if "CANDLE_DATA_DIR" in os.environ and "IMPROVE_DATA_DIR" in os.environ:
            if os.getenv('IMPROVE_DATA_DIR') != os.getenv("CANDLE_DATA_DIR"):
                self.logger.error("Found CANDLE_DATA_DIR and IMPROVE_DATA_DIR but not identical.")
                raise ValueError('Alias not identical')
            else:
                self.config.set("DEFAULT", "input_dir", os.getenv("IMPROVE_DATA_DIR", "./"))

        elif "CANDLE_DATA_DIR" in os.environ:
            self.logger.debug("Setting IMPROVE_DATA_DIR to CANDLE_DATA_DIR")
            os.environ["IMPROVE_DATA_DIR"] = os.environ["CANDLE_DATA_DIR"]

        if "IMPROVE_OUTPUT_DIR" not in os.environ:
            self.logger.debug('Setting IMPROVE_OUTPUT_DIR to IMPROVE_DATA_DIR or default')
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
        """Loads the configuration from a file.

        This method attempts to read the configuration settings from a specified
        file. If the file exists and is accessible, the configuration is loaded
        into the `config` attribute. If the file does not exist or cannot be
        accessed, an error is logged and the `DEFAULT` section of the configuration
        is initialized as an empty dictionary.
        """
        # Check if the file path is set and the file exists
        if self.file and os.path.isfile(self.file):
            # Log the action of loading the configuration
            self.logger.info("Loading config from %s", self.file)
            # Read the configuration file into the config attribute
            self.config.read(self.file)
        else:
            # Log an error if the file cannot be loaded
            self.logger.error("Can't load config from %s", str(self.file))
            # Initialize the DEFAULT section as an empty dictionary
            self.config['DEFAULT'] = {}

            
    def ini2dict(self, section: Optional[str] = None, flat: bool = False) -> dict:
        """Converts INI configuration to a dictionary.

        This method returns a dictionary representation of the configuration
        options. If a specific section is provided, it returns the options
        within that section. If `flat` is set to True, it returns a flat
        dictionary without sections, combining all options.

        Args:
            section (Optional[str]): The section of the configuration to convert.
                If None, all sections are included.
            flat (bool): If True, returns a flat dictionary without sections.

        Returns:
            dict: A dictionary containing the configuration options.
        """
        params = {}

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


    def dict(self, section: Optional[str] = None) -> dict:
        """Returns a dictionary of configuration options.

        This method serves as an alias for `ini2dict`, providing a dictionary
        representation of the configuration options. If a specific section is
        provided, it returns the options within that section.

        Args:
            section (Optional[str]): The section of the configuration to convert.
                If None, all sections are included.

        Returns:
            dict: A dictionary containing the configuration options for the specified
            section, or all sections if no section is specified.
        """
        return self.ini2dict(section=section)


    def save_parameter_file(self, file_name: Optional[str]) -> None:
        """Saves the final parameters to a file.

        This method writes the current parameters to a specified file. If the
        file name is an absolute path, it saves directly to that location.
        Otherwise, it saves the file in the `output_dir`. If the directory
        does not exist, it is created.

        Args:
            file_name (Optional[str]): The name of the file to save the parameters to.

        Raises:
            IOError: If there is an error writing the parameters to the file.
        """
        if file_name is None:
            # Log a warning if no file name is provided
            self.logger.warning("No file name provided to save parameters.")
            return

        # Log the action of saving parameters
        self.logger.debug("Saving parameters to %s", file_name)
        if os.path.isabs(file_name):
            # Use the absolute path if provided
            path = file_name
        else:
            # Construct the path in the output directory
            path = Path(self.output_dir, file_name)
            # Create the directory if it does not exist
            if not Path(path.parent).exists():
                self.logger.debug("Creating directory %s for saving config file.", path.parent)
                Path(path.parent).mkdir(parents=True, exist_ok=True)

        try:
            # Write the parameters to the file
            with path.open("w") as f:
                f.write(str(self.params))
        except IOError as e:
            # Log an error if file writing fails
            self.logger.error("Failed to save parameters to %s: %s", path, e)



            
    # ==========================================================
    # COMMAND LINE INTERFACE METHODS
    # These methods manage the parsing and handling of command line arguments.
    # They set up command line options, retrieve user inputs, and update defaults,
    # enabling dynamic configuration of the application via the command line.
    # ==========================================================
    def set_command_line_options(self, options: Optional[list] = None, group: Optional[str] = None) -> bool:
        """Set command line options using the CLI class.

        This function delegates the setup of command line options to the CLI class,
        ensuring that options are properly configured and integrated with the
        application's configuration management system. After setting the options,
        it updates the internal `_options` dictionary to reflect these changes,
        ensuring that all command-line options are tracked and managed.

        Args:
            options (list): A list of dictionaries defining command line options.
            group (str, optional): The name of the argument group to add options to.

        Returns:
            bool: True if the command line options were successfully set.
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
    

    def get_command_line_options(self) -> dict:
        """Retrieve command line options.

        This function updates the command line defaults with the current configuration
        by calling `_update_cli_defaults`, ensuring that any changes in the configuration
        are reflected in the command line options. It then retrieves the parsed command
        line arguments using the CLI class.

        Returns:
            dict: A dictionary containing the parsed command line options.
        """
        try:
            self._update_cli_defaults()
            return self.cli.get_command_line_options()
        except Exception as e:
            self.logger.error("Failed to retrieve command line options: %s", e)
            return {}
    
    
    def load_cli_parameters(self, file: str, section: str = None) -> dict:
        """Loads command line parameters from a file.

        This function reads parameter definitions from a specified file, which can be in JSON or YAML format.
        It validates the parameters to ensure they meet expected criteria and returns them as a dictionary.

        Args:
            file (str): The path to the file containing parameter definitions.
            section (str, optional): The section of the file to load parameters from, if applicable.

        Returns:
            dict: A dictionary containing the loaded parameters, where each key is a parameter name
            and the value is the parameter's configuration.

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
    
    
    def update_defaults(self, cli_definitions: list = None, new_defaults: dict = None) -> list:
        """Update the default values for command line arguments.

        This function updates the default values for command line arguments based on
        new defaults provided. It modifies the command line definitions and updates
        the parser's defaults if the options already exist.

        Args:
            cli_definitions (list): A list of dictionaries defining command line options.
            new_defaults (dict): A dictionary containing new default values for the options.

        Returns:
            list: A list of updated command line definitions with new default values.

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


    def update_cli_definitions(self, definitions: list = None) -> list:
        """Update the command line argument definitions with values from the config file.

        This function extracts the config file name from command line arguments and loads
        the config file. It then updates the provided command line argument definitions
        with values from the config file. This should be used before calling
        `self.cli.set_command_line_options(options=updated_parameters)`.

        Args:
            definitions (list, optional): A list of dictionaries defining command line options
                to be updated with values from the config file.

        Returns:
            list: A list of updated command line definitions with values from the config file.
        """
        # Config file can be provided as a command line argument or as a default in the code
        # Get the config file from the command line arguments otherwise use the default from self.file
        config_file_from_cli = self.cli.get_config_file()
        
        # Set self.file; the config will be loaded from self.file
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
    
    
    def _add_option(self, name: str, option: dict) -> bool:
        """Adds a command line option definition to the internal _options dictionary.

        This method checks the validity of the option definition, ensuring it is a
        dictionary with the required keys and that the name matches the expected
        values. It also verifies that the type and default values are supported.
        If the option is valid, it is added to the internal _options dictionary.

        Args:
            name (str): The name of the command line option.
            option (dict): A dictionary defining the command line option, including
                keys such as 'name', 'type', 'default', and 'help'.

        Returns:
            bool: True if the option was successfully added, False if the option
            was already defined.

        Raises:
            SystemExit: If the option is not a dictionary, if the name does not
            match the expected values, or if the type is unsupported.
        """
        # Check if option is a dictionary
        if not isinstance(option, dict):
            self.logger.error("Option %s is not a dictionary", name)
            sys.exit(1)
        
        # Check if name is identical to the name in the dictionary
        if "name" in option:
            if not name == option['name']:
                self.logger.error("Option name %s is not identical to name in dictionary %s", name, option['name'])
                sys.exit(1)
        elif not name == option['dest']:
            self.logger.error("Option name %s is not identical to name in dictionary %s", name, option['dest'])
            sys.exit(1)

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
        supported_types = {str, int, float, bool, str2bool, 'str', 'int', 'float', 'bool', 'str2bool', None}
        if option['type'] not in supported_types:
            self.logger.error("Unsupported type %s for option %s", option['type'], name)
            sys.exit(1)

        # Add option to _options    
        self._options[name] = option
        return True
    
    
    def _update_options(self) -> bool:
        """Update internal options with command line arguments.

        This function updates the internal `_options` dictionary with the current
        command line arguments parsed by the CLI class. It should be called every
        time a new option is added to the command line, such as after calling
        `set_command_line_options`.

        Returns:
            bool: True if all options were successfully updated, False if any option
            failed to be added.
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
        """Update command line defaults with values from the configuration file.

        This function updates the command line defaults in the CLI parser with values
        from the configuration file. It reads the configuration file specified by the
        command line or a default file and updates the defaults for each option.

        Returns:
            bool: True if the defaults were successfully updated.
        """
        # Attempt to retrieve the config file from command line arguments
        config_file = self.cli.get_config_file()

        # If a config file is specified via command line, use it; otherwise, use the default
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
                for option in self.config.items(section):
                    key, value = option
                    # Check if the option is defined in the internal _options dictionary
                    if key in self._options:
                        # If the option expects a list, attempt to parse the value as JSON
                        if 'nargs' in self._options[key] and \
                                self._options[key]['nargs'] and \
                                self._options[key]['nargs'] not in [None, 0, 1, "0", "1"]:
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                self.logger.error("Can't convert %s to list", value)
                                raise ValueError(f"Invalid JSON format for {key}: {value}")
                        # If a specific type is set for the option, convert the value to that type
                        elif 'type' in self._options[key]:
                            t = self._options[key]['type']
                            if t == 'str' or t == str:
                                value = str(value)
                            elif t == 'int' or t == int:
                                value = int(value)
                            elif t == 'float' or t == float:
                                value = float(value)
                            elif t == 'bool' or t == bool:
                                value = str2bool(value)
                            elif t == 'str2bool':
                                value = str2bool(value)
                            else:
                                self.logger.error("Unsupported type %s", self._options[key]['type'])
                                value = str(value)

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
    
    
 

    def get_param(self, section="DEFAULT", key=None) -> str:
        """Retrieves the value for a given configuration option.

        This method retrieves the value of a configuration option within a specified
        section. If no section is provided, the 'DEFAULT' section is used. Allowed
        section names are: 'Preprocess', 'Train', and 'Infer'.

        Args:
            section (str): The section of the configuration to access. Defaults to 'DEFAULT'.
            key (str): The key of the configuration option.

        Returns:
            str: The value of the configuration option.

        Raises:
            ValueError: If the key is not provided or the option is not found in the section.
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


    def set_param(self, section="DEFAULT", key=None, value=None) -> (str, str):
        """Sets a value for a given configuration option.

        This method sets the value of a configuration option within a specified
        section. If no section is provided, the 'DEFAULT' section is used. Allowed
        section names are: 'Preprocess', 'Train', and 'Infer'.

        Args:
            section (str): The section of the configuration to access. Defaults to 'DEFAULT'.
            key (str): The key of the configuration option.
            value (str, optional): The value to set for the given key. If None, an empty string is set.

        Returns:
            tuple: A tuple containing the value of the configuration option and a message.
                If the operation is successful, the message will be None.

        Raises:
            ValueError: If the key is not provided.
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
    
    



    def _validate_parameters(self, params: Optional[List[Dict[str, any]]], required: Optional[List[str]] = None) -> None:
        """Validates and sets types for configuration parameters.

        This method checks each parameter in the provided list to ensure it has a valid
        type and converts it to the corresponding Python type. It also checks for any
        required parameters if specified.

        Args:
            params (list of dict): A list of parameter dictionaries to validate. Each
                dictionary should contain a 'type' key indicating the expected type.
            required (list of str, optional): A list of required parameter names. If
                provided, the method checks that these parameters are present in the
                params list.

        Returns:
            None

        Raises:
            ValueError: If a required parameter is missing or if an unsupported type is encountered.
        """
        # Return early if no parameters are provided
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


    def load_parameter_definitions(self, file, section=None):
        """
        Load parameters definitions from from a file. 
        Used if not passed as a list.
        """
        self.logger.debug("Loading parameters from %s", file)

        # Convert Path to string
        if file and isinstance(file, Path):
            file = str(file)

        if os.path.isfile(file):
            # check if yaml or json file and load
            params = None

            if file.endswith('.json'):
                with open(file, 'r') as f:
                    params = json.load(f)
            elif file.endswith('.yaml') or file.endswith('.yml'):
                with open(file, 'r') as f:
                    params = yaml.safe_load(f)
            else:
                self.logger.error("Unsupported file format")
            self._validate_parameters(params)
            return params
        else:
            print(isinstance(file, str))
            self.logger.critical("Can't find file %s", file)
            sys.exit(1)
            return None



    

    def section_parameters(self, section=None) -> dict:
        """
        Return a dictionary of all options in the config file. If section
        is provided, return a dictionary of options in that section.
        TODO do really want of overload python's dict function?
        """
    
        params = {}
        sections = []

        if section:
            sections = [section]
        else:
            sections = self.config.sections()

        if section:
            # check if section exists
            if section in self.config:
                for i in self.config.items(section):
                    params[i[0]] = i[1]
            else:
                self.logger.error("Can't find section %s", section)

        else:
            for s in self.config.sections():
                params[s] = {}
                for i in self.config.items(s):
                    params[s][i[0]] = i[1]

        return params
    
    
    def initialize_parameters(self,
                              pathToModelDir,
                              section='DEFAULT',
                              default_config=None,  # located in ModelDir
                              additional_definitions=None,
                              required=None,):
        """Initialize parameters from command line and config file."""
        self.logger.setLevel(self.log_level) #why
        self.logger.debug("Initializing parameters for %s", section)
        # preserve the type of the object
        current_class = self.__class__
        self.__class__ = Config

        # Set section - DEFAULT, Preprocess, Train, Infer - maybe move to init
        # section is needed for reading config file
        self.section = section

        # Check if default config file is provided and reachable
        if default_config:
            if pathToModelDir:
                # check if type string or Path
                if not isinstance(pathToModelDir, Path):
                    pathToModelDir = Path(pathToModelDir)

                if not default_config.startswith("/"):
                    default_config = pathToModelDir / default_config
                else:
                    self.logger.error("No path to model directory provided.")
            if not os.path.isfile(default_config):
                self.logger.error("Can't find default config file %s", default_config)
                sys.exit(404)
            else:
                self.logger.debug("Default config file found: %s", default_config)
                self.file = default_config
        else:
            self.logger.warning("No default config file provided.")



        # Set and get command line args
        #
        # additonal_definitions in argparse format plus name:
        # [{ 'action' : 'store' , 'choices' : [ 'A' , 'B' , 'C' ] , 'type' : str , 'name' : "dest" }]
        # use set_set_command_line_options or cli.parser.add_argument(....)

        # Find duplicate dicts in additon_definitions for the key 'name'
        # if in dict then remove and log warning
        
        ### Set and get command line args

        # Update definitions with values from config file
        updated_definitions = None

        # set file to default_config if provided. load_config will use it if not specified on the command line
        # if default_config:
        #     self.file = default_config
        if additional_definitions:
            self.logger.debug("Updating additional definitions with values from config file.")
            # updated_definitions = self.update_cli_definitions(definitions=additional_definitions)
        else:
            self.logger.debug("No additional definitions provided.")
            sys.exit(0)
            updated_definitions = additional_definitions
        # Set command line options
        self.set_command_line_options(options=additional_definitions)
        # Get command line options
        self.params = self.get_command_line_options()
        # self.params=self.cli.get_command_line_options()
        # Set input and output directories
        self.input_dir = self.cli.args.input_dir
        self.output_dir = self.cli.args.output_dir
        self.log_level = self.cli.args.log_level
        self.logger.setLevel(self.log_level)
        self.logger.debug("Current log level is %s", self.log_level)

        # Set log level
        if "log_level" in self.cli.params:
            self.logger.info("Log level set by command line, updating to %s",
                             self.cli.params["log_level"])
            self.log_level = self.params["log_level"]
            self.logger.setLevel(self.log_level)

        self.logger.debug("Final parameters: %s", self.cli.cli_params)
        self.logger.debug("Final parameters: %s", self.params)
        self.logger.debug("Final parameters set.")

        # Set supported environment variables
        os.environ["IMPROVE_DATA_DIR"] = self.input_dir
        os.environ["IMPROVE_OUTPUT_DIR"] = self.output_dir
        os.environ["IMPROVE_LOG_LEVEL"] = self.log_level

        # Create output directory if not exists
        if not os.path.isdir(self.output_dir):
            self.logger.debug("Creating output directory: %s", self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        # Save parameters to file
        self.logger.debug("Saving final parameters to file.")
        # Save final configuration to file

        final_config_file = None
        if "param_log_file" in self.params:
            final_config_file=self.params["param_log_file"]
            self.save_parameter_file(final_config_file)

        self.__class__ = current_class
        return self.params
    
    
    
if __name__ == "__main__":
    # ==========================================================
    # TEST/DEBUGGING BLOCK
    # This section is used for testing and debugging the Config class functionality.
    # It demonstrates how to initialize the Config class, load parameters, and
    # interact with command line options.
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
    params = cfg.load_cli_parameters(test_dir / "data/additional_command_line_parameters.yml")
    print("Loaded CLI Parameters:", params)

    # Set up argparse for testing command line options
    import argparse
    cfg_parser = argparse.ArgumentParser(description='Get the config file from command line.', add_help=False)
    cfg_parser.add_argument('--config_file', metavar='INI_FILE', type=str, dest="config_file")

    # Simulate command line arguments for testing
    sys.argv.extend(["--config_file", str(test_dir / "data/default.cfg")])

    # Add a test command line option
    cfg.cli.parser.add_argument('--test', metavar='TEST_COMMAND_LINE_OPTION', dest="test",
                                nargs='+', type=int, default=[1], help="Test command line option.")

    # Initialize parameters with common and additional definitions
    final_params = cfg.initialize_parameters("./", additional_definitions=common_parameters + params)
    print("Initialized Parameters:", final_params)

    # Output the results to verify correct processing and storage of parameters
    print("Config Items in 'DEFAULT':", cfg.config.items('DEFAULT', raw=False))
    print("Parsed CLI Arguments:", cfg.cli.args)
    print("Final Parameters:", cfg.params)

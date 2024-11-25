import os
import sys
import logging
import configparser
import yaml
import json
from pathlib import Path

from improvelib.utils import str2bool, cast_value
from improvelib.initializer.cli import CLI

BREAK=os.getenv("IMPROVE_DEV_DEBUG", None)


class Config:
    """Class to handle configuration files."""
    # ConfigError = str
    config_sections = ['DEFAULT', 'Preprocess', 'Train', 'Infer']

    def __init__(self) -> None:

        # Default format
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        required = ["input_dir", "output_dir", "log_level", 'config_file']

        self.params = {}
        self.file = None # change to config_file
        self.logger = logging.getLogger('Config')
        self.log_level = os.getenv("IMPROVE_LOG_LEVEL", logging.INFO)
        self.logger.setLevel(self.log_level)

        self.required = required
        self.config = configparser.ConfigParser()
        self.cli = CLI()
        # Default values are set in command line parser
        self.input_dir = None
        self.output_dir = None
        self.params = {}
        self._options = {}

        # Set Defaults and conventions
        if "CANDLE_DATA_DIR" in os.environ and "IMPROVE_DATA_DIR" in os.environ:
            if not os.getenv('IMPROVE_DATA_DIR') == os.getenv("CANDLE_DATA_DIR"):
                self.logger.error(
                    "Found CANDLE_DATA_DIR and IMPROVE_DATA_DIR but not identical.")
                raise ValueError('Alias not identical')
            else:
                self.config.set("DEFAULT", "input_dir",
                                os.getenv("IMPROVE_DATA_DIR", "./"))

        elif "CANDLE_DATA_DIR" in os.environ:
            os.environ["IMPROVE_DATA_DIR"] = os.environ["CANDLE_DATA_DIR"]
        else:
            pass

        if not "IMPOVE_OUTPUT_DIR" in os.environ:
            self.logger.debug('Setting output directory')
            os.environ["IMPROVE_OUTPUT_DIR"] = os.environ.get(
                "IMPROVE_DATA_DIR", "./")

        self.config.set("DEFAULT", "input_dir",
                        os.environ.get("IMPROVE_DATA_DIR", "./"))
        self.config.set("DEFAULT", "output_dir",
                        os.environ.get("IMPROVE_OUTPUT_DIR", "./"))


    # Method to add a command line option definition to _options
    # This is used to check type and default values later
    def _add_option(self, name, option):

        # check if option is a dictionary
        if not isinstance(option, dict):
            self.logger.error("Option %s is not a dictionary", name)
            sys.exit(1)
        
        # check if name is identical to the name in the dictionary
        if "name" in option:
            if not name == option['name']:
                self.logger.error("Option name %s is not identical to name in dictionary %s", name, option['name'])
                sys.exit(1)
        elif not name == option['dest']:
            self.logger.error("Option name %s is not identical to name in dictionary %s", name, option['dest'])
            sys.exit(1)
        else:
            option['name'] = option['dest']

        # check if name is already in _options
        if name in self._options:
            self.logger.error("Option %s is already defined. Skipping.", name)
            return False

        # check if all required keys are present
        if not all(k in option for k in ('name', 'type', 'default', 'help')):
            self.logger.warning("Option %s is missing required keys.", name)
            self.logger.debug(option)

        # check if type and default are supported 
        if "type" not in option:
            self.logger.error("Option %s is missing type. Setting to str.", name)
            option['type'] = str
        if "default" not in option:
            self.logger.error("Option %s is missing default. Setting to None.", name)
            option['default'] = None
            
        if not option['type'] in ['str', 'int', 'float', 'bool', 'str2bool', None, str , int, float, bool, str2bool]:
            self.logger.error("Unsupported type %s for option %s", option['type'], name)
            sys.exit(1)

        # add option to _options    
        self._options[name] = option
        return True
    
    
    # Update _options with options from the command line (argparse)
    # Call everytime a new option is added to the command line, e.g. after set_command_line_options
    def _update_options(self):
        for action in self.cli.parser._actions:
            self._add_option(action.dest, action.__dict__)
        return True       
    
    # Update command line defaults with values from _options
    def _update_cli_defaults(self):

        # Read config
        
        # Config file can be provided as a command line argument or as a default in the code
        # Get the config file from the command line arguments otherwise use the default from self.file
        config_file = self.cli.get_config_file()
        
            
        # Set self.file ; the config will be loaded from self.file
        if config_file is not None:
            self.file = config_file
        else:
            self.logger.debug("No config file provided in command line arguments.")    

        if self.file is None:
            self.logger.debug("No config file provided at all.")
            return
        
        # Load the config file
        self.load_config()
        
        # Loop through config file and update command line defaults
        for section in self.config.sections():
            if self.section is None or self.section == section:
                for option in self.config.items(section):
                    self.logger.debug(f"Found {option} in ini file")
                    (key, value) = option
                    if key in self._options:

                        # check if type is set and cast option[1] to python type
                        if 'nargs' in self._options[key] and \
                            self._options[key]['nargs'] and \
                            self._options[key]['nargs'] not in [None, 0, 1 , "0", "1"]:
                            if BREAK :
                                breakpoint()
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                self.logger.error("Can't convert %s to list", value)
                                self.logger.critical(json.JSONDecodeError)
                                sys.exit(1)
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
                                self.logger.error("Unsupported type %s",
                                                  self._options[option[0]]['type'])
                                value = str(value)

                        self.cli.parser.set_defaults(**{key: value})

        return True


    def set_command_line_options(self, options=[] , group=None):
        """Set command line options."""
        self.cli.set_command_line_options(options) 
        self._update_options()
        return True
    

    def get_command_line_options(self):
        """Get command line options."""
        self._update_cli_defaults()
        return self.cli.get_command_line_options()


    def load_config(self):
        """ TODO ... """
        if self.file and os.path.isfile(self.file):
            self.logger.info("Loading config from %s", self.file)
            # if ini file
            if self.file.endswith('.ini'):
                self.config.read(self.file)
            # if yaml file
            elif self.file.endswith('.yaml') or self.file.endswith('.yml'):
                with open(self.file, 'r') as f:
                    self.config = yaml.safe_load(f)
            # if json file
            elif self.file.endswith('.json'):
                with open(self.file, 'r') as f:
                    self.config = json.load(f)
        else:
            self.logger.error("Not a file %s", str(self.file))
            self.config['DEFAULT'] = {}


    def save_parameter_file(self, file_name):
        """ 
        Saves final parameters to a file. 
        Saves in output_dir if file name given or anywhere with absolute path
        TODO: file name needs to be a general parameter (see initialize_param crazy name)
        TODO: would be nice to specifiy output format
        """
        if file_name is None:
            self.logger.warning("No file name provided to save parameters.")
            return
        else:
            self.logger.debug("Saving parameters to %s", file_name)
            if os.path.isabs(file_name):
                path = file_name
            else:
                path = Path(self.output_dir, file_name)
                if not Path(path.parent).exists():
                    self.logger.debug(
                        "Creating directory %s for saving config file.", path.parent)
                    Path(path.parent).mkdir(parents=True, exist_ok=True)

            with path.open("w") as f:
                f.write(str(self.params)) 


    def save_config(self, file_name, config=None):
        if os.path.isabs(file_name):
            with open(file_name, 'w') as out_file:
                self.config.write(out_file)
        else:
            path = Path(self.output_dir, file_name)
            if not Path(path.parent).exists():
                self.logger.debug(
                    "Creating directory %s for saving config file.", path.parent)
                Path(path.parent).mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            f.write(str(self.params)) 
      

    def param(self, section="DEFAULT" , key=None , value=None) -> (str,str):
        """
        Get or set value for given option. Gets or sets value in DEFAULT section
        if section is not provided. 
        Allowed section names are: Preprocess, Train and Infer
        """
        
        error=None

        if value is not None:
            if self.config.has_section(section):
                self.config[section][key]=value
            else:
                error="Unknow section " + str(section)
                self.logger.error(error)

        if self.config.has_option(section, key):
            value=self.config[section][key]
        else:
            error="Can't find option " + str(key)
            self.logger.error(error)
            value=None

        return (value, error)


    def get_param(self, section="DEFAULT", key=None) -> str:
        """
        Get value for given option. Gets or sets value in DEFAULT section if section is not provided. 
        Allowed section names are: Preprocess, Train and Infer
        """

        error = None

        if self.config.has_option(section, key):
            value = self.config[section][key]
        else:
            error = "Can't find option " + str(key)
            self.logger.error(error)
            value = None

        return value


    def set_param(self, section="DEFAULT", key=None, value=None) -> (str, str):
        """
        Set value for given option. Gets or sets value in DEFAULT section if section is not provided. 
        Allowed section names are: Preprocess, Train and Infer
        """
    
        msg = None

        if key:
            if not self.config.has_section(section) and not section == "DEFAULT":
                msg = "Unknow section " + str(section)
                self.logger.debug(msg)
                self.config[section] = {}

            if value is None:
                value = ''

            self.logger.debug("Key:%s\tValue:%s", key, value)
            self.config[section][key] = str(value)

        else:
            msg = "Can't update config, empty key"
            self.logger.error(msg)
            return (None, msg)

        return (self.config[section][key], msg)
    

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
    

    def check_required(self):
        """Check if all required parameters are set."""
        pass


    def _validate_parameters(self, params, required=None):
        """Validate parameters. Set types and check for required parameters."""

        if params is None:
            return

        for p in params:
            # check if type is set and convert to python type
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
                    self.logger.error("Unsupported type %s", p['type'])
                    p['type'] = str


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


    def validate_parameters(self, params, required=None):
        """Validate parameters."""
        pass


    def load_config_file(self, pathToModelDir=None, default_config=None):
        """
        Loads the configuration file. 
        """
        if self.input_dir and os.path.isdir(self.input_dir):

            # Set config file name
            if self.cli.args.config_file:
                self.file = self.cli.args.config_file
            else:
                # Make pathToModelDir and default_config same type. Possible types are: str, Path
                if isinstance(pathToModelDir, Path):
                    pathToModelDir = str(pathToModelDir)
                if isinstance(default_config, Path):
                    default_config = str(default_config)

                if pathToModelDir is not None:
                    if not pathToModelDir.endswith("/"):
                        pathToModelDir += "/"
                else:
                    pathToModelDir = "./"

                if default_config is not None:
                    if not os.path.abspath(default_config):
                        self.logger.debug(
                            "Not absolute path for config file. Should be \
                            relative to model directory")
                        self.file = pathToModelDir + default_config
                    else:
                        self.logger.warning("Default config not releative to \
                                            model directory. Using as is.")
                        self.file = default_config
                        
                else:
                    self.logger.error("No default config file provided")

                self.logger.debug("No config file provided. Using default: %s", self.file)

            # Set full path for config
            if self.file and not os.path.abspath(self.file):
                self.logger.debug(
                    "Not absolute path for config file. Should be relative to input_dir")
                self.file = self.input_dir + "/" + self.file
                self.logger.debug("New file path: %s", self.file)

            # Load config if file exists
            if self.file and os.path.isfile(self.file):
                self.load_config()
            else:
                self.logger.warning("Can't find config file: %s", self.file)
                self.config[section] = {}
        else:
            self.logger.critical("No input directory: %s", self.input_dir)


    def ini2dict(self, section=None , flat=False) -> dict:
        """
        Return a dictionary of all options in the config file. If section is provided,
        return a dictionary of options in that section. If flat is True, return a flat
        dictionary without sections.
        """

        params = {}
        sections=[]

        if section :
            sections=[section]
        else:
            sections=self.config.sections()
        
        if section:
            # check if section exists
            if self.config.has_section(section):
                for i in self.config.items(section):
                    params[i[0]]=i[1]
            else:
                self.logger.error("Can't find section %s", section)

        else:
            if flat:
                for s in self.config.sections():
                    for i in self.config.items(s):
                        params[i[0]]=i[1]
            else:
                for s in self.config.sections():
                    params[s]={}
                    for i in self.config.items(s):
                        params[s][i[0]]=i[1]

        return params


    def dict(self, section=None) -> dict : # rename to ini2dict ; keep dict as alias
        """
        Return a dictionary of all options in the config file. If section is provided,
        return a dictionary of options in that section
        """
        return self.ini2dict(section=section)


    # Load command line definitions from a file
    def load_cli_parameters(self, file, section=None):
        """Load parameters from a file."""
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


    # Update the default values for the command line arguments with the new defaults
    def update_defaults(self, cli_definitions=None, new_defaults=None):

        # Get the list of added options from the parser
        existing_options = [o.lstrip('-')
                              for o in self.cli.parser._option_string_actions]

        if not new_defaults:
            self.logger.error("No new defaults provided.")
            return
        if not cli_definitions:
            self.logger.error("No command line definitions provided.")
            return

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

    # Extract config file name from command line arguments and load config file
    # Seed defaults for command line arguments with values from config file

    def update_cli_definitions(self, definitions=None):
        """
        Update the command line argument definitions with values from the config file.
        Use this before self.cli.set_command_line_options(options=updated_parameters)
        """

        # Config file can be provided as a command line argument or as a default in the code
        # Get the config file from the command line arguments otherwise use the default from self.file
        config_file_from_cli = self.cli.get_config_file()
        
            
        # Set self.file ; the config will be loaded from self.file
        if config_file_from_cli is not None:
            self.file = config_file_from_cli
        else:
            self.logger.debug("No config file provided in command line arguments.")    

        if self.file is None:
            self.logger.debug("No config file provided at all.")
            return
        
        # Load the config file
        self.load_config()
        
        # Update additional_definitions with values from config file
        return self.update_defaults(cli_definitions=definitions, new_defaults=self.ini2dict(flat=True))
   

    def initialize_parameters(self,
                              pathToModelDir,
                              section='DEFAULT',
                              default_config=None,  # located in ModelDir
                              default_model=None,
                              additional_definitions=None,
                              required=None,):
        """Initialize parameters from command line and config file."""
        self.logger.setLevel(self.log_level) #why
        self.logger.debug("Initializing parameters for %s", section)
        # preserve the type of the object
        current_class = self.__class__
        self.__class__ = Config

        # Set section - DEFAULT, Preprocess, Train, Infer - maybe move to init
        # section is neeed for reading config file
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
        # if additional_definitions:
        #     self.logger.debug("Updating additional definitions with values from config file.")
        #     # updated_definitions = self.update_cli_definitions(definitions=additional_definitions)
        # else:
        #     self.logger.debug("No additional definitions provided.")
        #     sys.exit(0)
        #     updated_definitions = additional_definitions
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
            os.makedirs(self.output_dir)
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
    cfg = Config()
    # cfg.file = "./Tests/Data/default.cfg"
    # cfg.output_dir = "./tmp"
    # cfg.load_config()
    # print(cfg.params)
    # print(cfg.dict())
    # print(cfg.param(None, 'weights', None))
    # cfg.param('Infer', 'weights', 'default.weights')
    # for section in cfg.config.items():
    #     print(section)
    #     for item in cfg.config.items(section[0], raw=False):
    #         print(item)
    # print(cfg.param("Infer", 'weights', None))
    # print(cfg.dict('Infer'))
    # cfg.save_config("./tmp/saved.config", config=cfg.config['DEFAULT'])

    common_parameters = [
        {
            "name": "list_of_int",
            "dest": "loint",
            "help": "Need help to display default value",
            "nargs" :"+",
            "type" :  int,
            "default": [100],
            "section": "DEFAULT"
        },
        {
            "name": "list_of_strings",
            "dest": "lostr",
            "nargs" : "+",
            "type" :  str,
            "default": ['100'],
            "section": "DEFAULT"
        },
        {
            "name": "list_of_lists",
            "nargs" : "+",
            "metavar": "lol",
            "dest": "l",
            "action": "append",
            "type" :  str,
            "default": [[1,2,3],[4,5,6]],
            "section": "DEFAULT"
        },
    ]

    # create path from current directory, keep everything before improvelib
    current_dir = Path(__file__).resolve().parent
    test_dir = current_dir.parents[1] / "tests"

    params = cfg.load_cli_parameters(
        test_dir / "data/additional_command_line_parameters.yml")
    print(params)

    # updated additional_definitions with values from config file
    # cfg.cli.set_command_line_options(options=params)
    import argparse
    cfg_parser = argparse.ArgumentParser(
                    description='Get the config file from command line.',
                    add_help=False,)
    cfg_parser.add_argument('--config_file', metavar='INI_FILE', type=str , dest="config_file")
    # parse command line and grab config file
    sys.argv.append( "--config_file" )
    sys.argv.append( str( test_dir / "data/default.cfg") )
 
    cfg.cli.parser.add_argument('--test', metavar='TEST_COMMAND_LINE_OPTION', dest="test",
                                nargs='+',
                                type=int,
                                default=[1], help="Test command line option.")
    
    # cfg.cli.parser.set_defaults(test=100)
    
    print(
        cfg.initialize_parameters(
        "./", additional_definitions=common_parameters + params)
    )
    print(cfg.config.items('DEFAULT', raw=False))
    print(cfg.cli.args)
    print(cfg.params)

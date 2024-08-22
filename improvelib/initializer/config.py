import os
import sys
import logging
import configparser
import yaml
import json
from pathlib import Path

from improvelib.utils import str2bool, cast_value
from improvelib.initializer.cli import CLI


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
        self.file = None
        self.logger = logging.getLogger('Config')
        self.log_level = os.getenv("IMPROVE_LOG_LEVEL", logging.DEBUG)
        self.logger.setLevel(self.log_level)

        self.required = required
        self.config = configparser.ConfigParser()
        self.cli = CLI()
        # Default values are set in command line parser
        self.input_dir = None
        self.output_dir = None
        self.final_params = {}

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

    def load_config(self):
        """ TODO ... """
        if self.file and os.path.isfile(self.file):
            self.logger.info("Loading config from %s", self.file)
            self.config.read(self.file)
        else:
            self.logger.error("Can't load config from %s", str(self.file))
            self.config['DEFAULT'] = {}


    def save_parameter_file(self, file_name):
        """ 
        Saves final parameters to a file. 
        Saves in output_dir if file name given or anywhere with absolute path
        TODO: file name needs to be a general parameter (see initialize_param crazy name)
        TODO: would be nice to specifiy output format
        """
        if os.path.isabs(file_name):
            path = file_name
        else:
            path = Path(self.output_dir, file_name)
            if not Path(path.parent).exists():
                self.logger.debug(
                    "Creating directory %s for saving config file.", path.parent)
                Path(path.parent).mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            f.write(str(self.final_params)) 


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
            f.write(str(self.final_params)) 
      

    def param(self, section="DEFAULT" , key=None , value=None) -> (str,str):
        """
        Get or set value for given option. Gets or sets value in DEFAULT section if section is not provided. 
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

    def load_config_file(self, pathToModelDir, default_config):
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
                    self.file = pathToModelDir + default_config
                else:
                    self.logger.warning("No default config file provided")

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

    def dict(self, section=None) -> dict :
        """
        Return a dictionary of all options in the config file. If section is provided, 
        return a dictionary of options in that section
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
            for s in self.config.sections():
                params[s]={}
                for i in self.config.items(s):
                    params[s][i[0]]=i[1]

        return params

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

    def initialize_parameters(self,
                              pathToModelDir,
                              section='DEFAULT',
                              default_config='default.cfg',  # located in ModelDir
                              default_model=None,
                              additional_definitions=None,
                              required=None,):
        """Initialize parameters from command line and config file."""
        self.logger.setLevel(self.log_level) #why
        self.logger.debug("Initializing parameters for %s", section)
        # preserve the type of the object
        current_class = self.__class__
        self.__class__ = Config


        # Set and get command line args
        #
        # additonal_definitions in argparse format plus name:
        # [{ 'action' : 'store' , 'choices' : [ 'A' , 'B' , 'C' ] , 'type' : str , 'name' : "dest" }]
        # use set_set_command_line_options or cli.parser.add_argument(....)

        # Find duplicate dicts in additon_definitions for the key 'name'
        # if in dict then remove and log warning

        
        #print("additional_definitions:", additional_definitions)
        """
        if additional_definitions:
            
            # Convert Path to string
            if additional_definitions and isinstance(additional_definitions, Path):
                additional_definitions = str(additional_definitions)

            # check if additional_definitions is a string
            if isinstance(additional_definitions, str) and os.path.isfile(additional_definitions):
                self.logger.debug(
                    "Loading additional definitions from file %s", additional_definitions)
                additional_definitions = self.load_cli_parameters(
                    additional_definitions)

            duplicates = []
            names = []
            for i, v in enumerate(additional_definitions):
                # print(i,additional_definitions[i])
                if additional_definitions[i]['name'] in names:
                    self.logger.warning(
                        "Duplicate definition for %s", additional_definitions[i]['name'])
                    duplicates.append(i)
                else:
                    names.append(additional_definitions[i]['name'])
            # Loop through duplicates in reverse order and remove from additional_definitions
            for i in duplicates[::-1]:
                additional_definitions.pop(i)
        """
        #cli = self.cli
        self.cli.set_command_line_options()

        config_file = self.cli.args.parse_known_args(['--config_file'])
        self.load_config_file(pathToModelDir, default_config)

        self.cli.get_command_line_options()

        # Set log level
        if "log_level" in self.cli.cli_params:
            self.logger.debug("Log level set by command line, updating to %s", self.cli.cli_params["log_level"])
            self.log_level = self.cli.cli_params["log_level"]
            self.logger.setLevel(self.log_level)

        # Load config file
        self.logger.debug("Loading configuration file")
        self.load_config_file(pathToModelDir, default_config)
        # Sets dictionary of parameters with defaults
        self.final_params = self.cli.default_params
        # Gets dictionary of parameters from config for this section
        section_config = dict(self.config[section])
        self.logger.debug("Default parameters: %s", self.cli.default_params)
        self.logger.debug("Current section: %s", section)
        self.logger.debug("Current section config parameters: %s", section_config)
        self.logger.debug("Updating config")

        # Overrides dictionary of defaults with config params
        # breakpoint();
        # from pprint import pprint
        # pprint(self.cli.default_params)
        for cfp in section_config:
            if cfp in self.final_params: # TODO why we need this??
                self.logger.info("Overriding %s default with config value of %s", cfp, section_config[cfp])
                self.final_params[cfp] = cast_value(section_config[cfp])
            else:
                self.logger.warning("Config parameter %s is not defined, skipping.", cfp)
        self.logger.debug("Current section CLI set parameters: %s", self.cli.cli_params)
        # pprint(self.final_params)
        # breakpoint();

        # Overrides dictionary of defaults+config with CLI params
        for clip in self.cli.cli_params:
            self.logger.info("Overriding %s default or config with command line value of %s", clip, self.cli.cli_params[clip])
            self.final_params[clip] = self.cli.cli_params[clip]
        self.logger.debug("Final parameters: %s", self.final_params)
        self.logger.debug("Final parameters set.")

        # Update input and output directories
        self.output_dir = self.final_params['output_dir']
        self.input_dir = self.final_params['input_dir']
        self.log_level = self.final_params['log_level']
        self.logger.setLevel(self.log_level)
        self.logger.debug("Current log level is %s", self.log_level)

        # Set environment variables
        # TODO why input_dir overrides IMPROVE_DATA_DIR?
        # NCK: this is a good question, what are we doing with this?
        os.environ["IMPROVE_DATA_DIR"] = self.input_dir
        os.environ["IMPROVE_OUTPUT_DIR"] = self.output_dir
        os.environ["IMPROVE_LOG_LEVEL"] = self.log_level

        # Create output directory if not exists
        if not os.path.isdir(self.output_dir):
            self.logger.debug("Creating output directory: %s", self.output_dir)
            os.makedirs(self.output_dir)
        # Save parameters to file
        self.logger.debug("Saving final parameters to file.")
        self.save_parameter_file(self.final_params["param_log_file"])

        self.__class__ = current_class
        return self.final_params


if __name__ == "__main__":
    cfg = Config()
    cfg.file = "./Tests/Data/default.cfg"
    cfg.output_dir = "./tmp"
    cfg.load_config()
    print(cfg.params)
    print(cfg.dict())
    print(cfg.param(None, 'weights', None))
    cfg.param('Infer', 'weights', 'default.weights')
    for section in cfg.config.items():
        print(section)
        for item in cfg.config.items(section[0], raw=False):
            print(item)
    print(cfg.param("Infer", 'weights', None))
    print(cfg.dict('Infer'))
    cfg.save_config("./tmp/saved.config", config=cfg.config['DEFAULT'])

    common_parameters = [
        {
            "name": "chkpt",
            "dest": "checkpointing",
            "type" : bool ,
            "default": False,
            "help": "Flag to enable checkpointing",
            "section": "DEFAULT"
        },
        {
            "name": "list_of_int",
            "dest": "loint",
            # "nargs" :"+",
            "type" :  int,
            "default": 100,
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
            "name": "list",
            "metavar": "lOfStrings",
            "dest": "l",
            "nargs" : "+",
            "type" :  str,
            "default": '100',
            "section": "DEFAULT"
        },
        {
            "name": "test_string",
            "metavar": "lOfStrings",
            "dest": "l",
            "type" :  str,
            "default": 100,
            "section": "DEFAULT"
        }
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
 
    args_tmp = cfg_parser.parse_known_args()
    config_file = args_tmp[0].config_file
    cfg.logger.debug("Config file: %s", config_file)
    # print(config_file)

    # load config file
    cfg.file = config_file
    cfg.load_config()
    print(cfg.dict())

    source_dict = cfg.dict()
    target_dict = {}

    for section in  source_dict:
        print(section)
        for item in source_dict[section]:
            # print(item, source_dict[section][item])
            target_dict[item] = source_dict[section][item]

    print(target_dict)

    print(common_parameters)
    updated_parameters = []
    import json
    for entry in common_parameters:
        print(entry)
        # print(entry['name'])
        if entry['name'] in target_dict:
            entry['default'] = target_dict[entry['name']]
            if "nargs" in entry:
                # entry['default'] = str2bool(entry['default'])
                print(target_dict[entry['name']])
                entry['default'] = json.loads(target_dict[entry['name']])
        updated_parameters.append(entry)
    print(updated_parameters)

    cfg.cli.parser.add_argument('--test', metavar='TEST_COMMAND_LINE_OPTION', dest="test",
                                type=str,
                                default="a", help="Test command line option.")

    cfg.cli.set_command_line_options(options=updated_parameters)
    args = cfg.cli.parser.parse_args()
    print(args)
    # get type for chkpt from args
    print(type(args.chkpt))
    
    # cfg.initialize_parameters(
    #     "./", additional_definitions=common_parameters + params)
    # print(cfg.config.items('DEFAULT', raw=False))

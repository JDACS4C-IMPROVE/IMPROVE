import os
import logging
import configparser
import yaml
import json

from pathlib import Path
from improvelib.cli import CLI




class Config:
    """Class to handle configuration files."""
    # ConfigError = str

    def __init__(self) -> None:

        # Default format 
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)
       
        required=[ "input_dir", "output_dir", "log_level", 'config_file']
        
  

        self.params = {}
        self.file = None
        self.logger = logging.getLogger('Config')
        self.log_level = os.getenv("IMPROVE_LOG_LEVEL" , logging.DEBUG)
        self.logger.setLevel(self.log_level)
       
        self.required = required
        self.config = configparser.ConfigParser()
        self.cli = CLI()
        # Default values are set in command line parser
        self.input_dir = None
        self.output_dir = None

        # Set Defaults and conventions
        
        if "CANDLE_DATA_DIR" in os.environ and "IMPROVE_DATA_DIR" in os.environ:
            if not os.getenv('IMPROVE_DATA_DIR') == os.getenv("CANDLE_DATA_DIR"):
                self.logger.error("Found CANDLE_DATA_DIR and IMPROVE_DATA_DIR but not identical.")
                raise ValueError('Alias not identical')
            else:
                self.config.set("DEFAULT" , "input_dir" , os.getenv("IMPROVE_DATA_DIR","./"))

        elif "CANDLE_DATA_DIR" in os.environ:
            os.environ["IMPROVE_DATA_DIR"] = os.environ["CANDLE_DATA_DIR"] 
        else:
            pass

        if not "IMPOVE_OUTPUT_DIR" in os.environ :
            self.logger.debug('Setting output directory')
            os.environ["IMPROVE_OUTPUT_DIR"] = os.environ.get("IMPROVE_DATA_DIR" , "./")

        self.config.set("DEFAULT" , "input_dir" , os.environ.get("IMPROVE_DATA_DIR" , "./"))
        self.config.set("DEFAULT" , "output_dir" , os.environ.get("IMPROVE_OUTPUT_DIR" , "./"))


    
        
    def load_config(self):

        if self.file and os.path.isfile(self.file) :
            self.logger.info("Loading config from %s" , self.file)
            self.config.read(self.file)
        else:
            self.logger.error("Can't load config from %s", str(self.file))
            self.config['DEFAULT'] = {}

        

    def save_config(self, file , config=None):
        if os.path.isabs(file):
            self.config.write(file)
        else:
            if config and 'output_dir' in config:
                path = Path(config['output_dir'] , file)
            else: 
                path = Path(file)

            if not Path(path.parent).exists() :
                self.logger.debug("Creating directory %s for saving config file." , path.parent)
                Path(path.parent).mkdir(parents=True , exist_ok = True)
    

            with path.open("w") as f:
                self.config.write(f)

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
    
    def get_param(self, section="DEFAULT" , key=None) -> str:
        """
        Get value for given option. Gets or sets value in DEFAULT section if section is not provided. 
        Allowed section names are: Preprocess, Train and Infer
        """
        
        error=None

        if self.config.has_option(section, key):
            value=self.config[section][key]
        else:
            error="Can't find option " + str(key)
            self.logger.error(error)
            value=None

        return value  

    def set_param(self, section="DEFAULT" , key=None , value=None) -> (str,str):
        """
        Set value for given option. Gets or sets value in DEFAULT section if section is not provided. 
        Allowed section names are: Preprocess, Train and Infer
        """
        
        msg=None

        if key:
            if not self.config.has_section(section) and not section=="DEFAULT":
                msg="Unknow section " + str(section)
                self.logger.debug(msg)
                self.config[section] = {}      
            
            if value is None :
                value = ''
            
            self.logger.debug("Key:%s\tValue:%s", key , value)
            self.config[section][key]=str(value)
        else:
            msg = "Can't update config, empty key"
            self.logger.error(msg)
            return (None, msg)

        return (self.config[section][key], msg)

    def dict(self, section=None) -> dict :
        """Return a dictionary of all options in the config file. If section is provided, return a dictionary of options in that section"""
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
    
    def check_required(self):
        """Check if all required parameters are set."""
        pass

    def load_parameters(self, file , section=None):
        """Load parameters from a file."""
        self.logger.debug("Loading parameters from %s", file)
        
        if isinstance(file, str) and os.path.isfile(file):   
            # check if yaml or json file and load

            if file.endswith('.json'):
                with open(file, 'r') as f:
                    return json.load(f)
            elif file.endswith('.yaml') or file.endswith('.yml'):
                with open(file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error("Unsupported file format")
                return None
        else:
            self.logger.error("Can't find file %s", file)
            return None

    def validate_parameters(self, params, required=None):
        """Validate parameters."""
        pass

    def initialize_parameters(self,
                              pathToModelDir,
                              section='DEFAULT',
                              default_config='default.cfg', # located in ModelDir
                              default_model=None,
                              additional_definitions=None,
                              required=None,):
        """Initialize parameters from command line and config file."""
        
        # preserve the type of the object
        current_class = self.__class__
        self.__class__ = Config

        print( "initialize_parameters" + str(type(self)) )

        # Set and get command line args
        #
        # additonal_definitions in argparse format plus name:
        # [{ 'action' : 'store' , 'choices' : [ 'A' , 'B' , 'C' ] , 'type' : str , 'name' : "dest" }]
        # use set_set_command_line_options or cli.parser.add_argument(....)

        # Find duplicate dicts in additon_definitions for the key 'name'
        # if in dict then remove and log warning

        self.logger.debug("Initializing parameters for %s", section)

        if additional_definitions:

            # check if additional_definitions is a string 
            if isinstance(additional_definitions, str) and os.path.isfile(additional_definitions):
                self.logger.debug("Loading additional definitions from file %s", additional_definitions)
                additional_definitions = self.load_parameters(additional_definitions)

            duplicates = []
            names = []
            for i,v in enumerate(additional_definitions):
                # print(i,additional_definitions[i])
                if additional_definitions[i]['name'] in names:
                    self.logger.warning("Duplicate definition for %s", additional_definitions[i]['name'])
                    duplicates.append(i)
                else:
                    names.append(additional_definitions[i]['name'])
            # Loop through duplicates in reverse order and remove from additional_definitions
            for i in duplicates[::-1]:
                additional_definitions.pop(i)
                
        # cli=CLI()
        cli = self.cli
        cli.set_command_line_options(options=additional_definitions)
        cli.get_command_line_options()

        # Set log level
        if cli.args.log_level:
            self.logger.setLevel(cli.args.log_level)
            self.log_level = cli.args.log_level

        # Load Config
        if os.path.isdir(cli.args.input_dir) :

            # Set config file name
            if cli.args.config_file:        
                self.file = cli.args.config_file
            else:
                # Make pathToModelDir and default_config same type. Possible types are: str, Path
                if isinstance(pathToModelDir, Path):
                    pathToModelDir = str(pathToModelDir)
                if isinstance(default_config, Path):
                    default_config = str(default_config)

                if pathToModelDir is not None :
                    if pathToModelDir.endswith("/"):
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
                self.logger.debug("Not absolute path for config file. Should be relative to input_dir")
                self.file = self.input_dir + "/" + self.file
                self.logger.debug("New file path: %s", self.file)

            # Load config if file exists
            if self.file and os.path.isfile(self.file):
                self.load_config()
            else:
                self.logger.warning("Can't find config file: %s", self.file)
                self.config[section] = {}
            

        else:
            self.logger.critical("No input directory: %s", cli.args.input_dir)

        # Set/update config with arguments from command line
        self.logger.debug("Updating config")
        for k in cli.params :
            self.logger.debug("Setting %s to %s", k, cli.params[k])
            self.set_param(section, k, cli.params[k])
            # self.config[section][k] = cli.params[k]
        
        # Update input and output directories    
        self.output_dir = self.config[section]['output_dir']
        self.input_dir = self.config[section]['input_dir']
        self.log_level = self.config[section]['log_level']
        self.logger.setLevel(self.log_level)
        
        # Set environment variables 
    
        os.environ["IMPROVE_DATA_DIR"] = self.input_dir
        os.environ["IMPROVE_OUTPUT_DIR"] = self.output_dir
        os.environ["IMPROVE_LOG_LEVEL"] = self.config[section]['log_level']

     


        # Create output directory if not exists
        if not os.path.isdir(self.output_dir):
            self.logger.debug("Creating output directory: %s", self.output_dir)
            os.makedirs(self.output_dir)
    
        self.__class__ = current_class
        return self.dict()
    
    




if __name__ == "__main__":
    cfg=Config()
    cfg.file="./Tests/Data/default.cfg"
    cfg.load_config()
    print(cfg.params)
    print(cfg.dict())
    print(cfg.param( None , 'weights' , None))
    cfg.param('Infer' , 'weights' ,'default.weights')
    for section in cfg.config.items():
        print(section)
        for item in cfg.config.items(section[0], raw=False):
            print(item)
    print(cfg.param( "Infer" , 'weights' , None))
    print(cfg.dict('Infer'))
    cfg.save_config("./tmp/saved.config" , config=cfg.config['DEFAULT'])

    common_parameters=[ 
        { 
            "name" : "chkpt", 
            "dest" : "checkpointing",
            # "type" : "str2bool" ,
            "default" : False,
            "help" : "Flag to enable checkpointing",
            "section" : "DEFAULT"
            } 
        ]

    params = cfg.load_parameters("./Tests/Data/common_parameters.yml")
    cfg.cli.set_command_line_options(options=params)
    print(params)


    cfg.initialize_parameters("./", additional_definitions=common_parameters + params )
    print(cfg.config.items('DEFAULT', raw=False))

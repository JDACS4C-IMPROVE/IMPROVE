import configparser
import json
import logging
import os

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'

class Config():
    def __init__(self):
        self.option = None
        self.config = None
        self.config_file = None
        self.input_dir = None
        self.output_dir = None
        self.log_level = None
        self.logger = None
        self.option = None
        self.params = None
        self.args = None
        # Default format for logging
        logging.basicConfig(format=FORMAT)
        self.logger=logging.getLogger('Common.Config')
        

    def load_config(self, file):
        if file.endswith('.ini'):
            self.config = configparser.ConfigParser()
            self.config.read(file)
            #self.option = dict(config.items('DEFAULT'))
            self.option={}
            combined_options = {section: dict(config.items(section)) for section in config.sections()}
            self.option.update(combined_options)
        elif file.endswith('.json'):
            with open(file, 'r') as f:
                self.option = json.load(f)
        else:
            raise ValueError("Unsupported file format")
        
        for key in self.option.keys():
            # create attribute for each key
            for k in self.option[key].keys():
                setattr(self, k, self.option[key][k])
            if self.logger.level == logging.DEBUG:
                print(k, self.option[key][k])

        # if ini file load ini file else load json file
        pass

    def save_config(self, file):

        if file.endswith('.ini'):
            config = configparser.ConfigParser()
            config['DEFAULT'] = self.option
            with open(file, 'w') as f:
                config.write(f)
        elif file.endswith('.json'):
            with open(file, 'w') as f:
                json.dump(self.option, f)
        else:
            raise ValueError("Unsupported file format")
        

        pass

    def set_param(self, section, key, value):
        self.option[section][key] = value
        setattr(self, key, value)
        pass


    def get_param(self, section="DEFAULT" , key=None) -> str:
      
        error=None

        if self.config.has_option(section, key):
            value=self.config[section][key]
        else:
            error="Can't find option " + str(key)
            self.logger.error(error)
            value=None

        return value  


    def get_enviorment_variable(self):

        pass


    def initialize_parameters(self,
                              cli=None, # Command Line Interface of type CLI
                              section='DEFAULT',
                              config_file=None,
                              additional_definitions=None,
                              required=None,):
        """Merge parameters from command line and config file."""

        # Set log level


        if cli and cli.args.log_level:
            self.logger.debug("Setting log level to %s", cli.args.log_level)
            self.args = cli.args
            self.logger.setLevel(cli.args.log_level)
            self.log_level = cli.args.log_level

        # preserve the type of the object
        current_class = self.__class__
        self.__class__ = Config


        self.logger.debug("Initializing parameters for %s", section)

        # Check if config_file is set and file exists
        if config_file:
            self.logger.debug("Config file: %s", config_file)
            if os.path.isfile(config_file):
                self.config_file = config_file
            else:
                self.logger.critical("Can't find config file: %s", config_file)
                self.config_file = None

            # Load Config        
            self.load_config(self.config_file)
    
            # Set/update config with arguments from command line
            self.logger.debug("Updating config")

            # create dictionary of parameters from list of tuples. The list of tuples is the result of cli.args.user_specified 
            # which is a list of tuples of the form (name, value) where name is the name of the parameter and value is the value of the parameter
            user_provided_params = dict(cli.user_specified)
            self.logger.debug("User provided parameters: %s", user_provided_params)

            for k in cli.params :
                # Test if k is a valid parameter and has a value
                if self.get_param(section, k) is None:
                    self.logger.error("Invalid parameter: %s", k)
                
                self.set_param(section, k, cli.params[k])


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
    cfg = Config()
    cfg.logger.setLevel(logging.DEBUG)
    
    cfg.load_config('config.ini')
    cfg.save_config('config.json')
    cfg.save_config('config.ini')
    cfg.save_config('config.txt')
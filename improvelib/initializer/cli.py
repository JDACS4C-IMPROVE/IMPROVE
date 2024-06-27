import logging
import argparse
import os
import sys

from candle import parse_from_dictlist
# from improve import config as BaseConfig



class CLI:
    """Base Class for Command Line Options"""

    def __init__(self):

        # Default format for logging
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        # Class attributes and defautl values
        # Initialize parser
        self.parser=argparse.ArgumentParser(description='IMPROVE Command Line Parser')

        # Initialize logger
        self.logger=logging.getLogger('CLI')

        # Command line options after parsing, results of self.parser.parse_args()
        self.args = None # placeholder for args from argparse
        self.params = None # dict of args
        
        # Set logger level
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.DEBUG))
      

    """        # Set common options for all model scripts
        common_options = self.parser.add_argument_group('Standard Options')
        common_options.add_argument('-i', '--input_dir', metavar='DIR', type=str, dest="input_dir",
                                  default=os.getenv("IMPROVE_INPUT_DIR" , "./"), 
                                  help='Base directory for input data. Default is IMPROVE_DATA_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        common_options.add_argument('-o', '--output_dir', metavar='DIR', type=str, dest="output_dir",
                                  default=os.getenv("IMPROVE_OUTPUT_DIR" , "./"), 
                                  help='Base directory for output data. Default is IMPROVE_OUTPUT_DIR or if not specified current working directory. All additional relative output pathes will be placed into the base output directory.')
        common_options.add_argument('--log_level', metavar='LEVEL', type=str, dest="log_level", 
                                  choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
                                  default=os.getenv("IMPROVE_LOG_LEVEL", "WARNING"), help="Set log levels. Default is WARNING. Levels are:\
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET") 
        common_options.add_argument('-cfg', '--config_file', metavar='INI_FILE', dest="config_file", 
                                  type=str,
                                  default=None, help="Config file in INI format.")  """
                                  


    def set_command_line_options(self,options=[] , group=None):
        """Set Command Line Options, saveguard standard options."""
        self.logger.debug("Setting Command Line Options")

        if not options:
            self.logger.warning("No options provided. Ignoring.")
            return

        predefined_options = [ o.lstrip('-') for o in self.parser._option_string_actions ]
        # ['input_dir', 'output_dir', 'log_level', 'config_file'] 
        for o in predefined_options:
            # check if o is the value of name in one of the dicts in options
            for d in options:
                if o == d['name']:
                    self.logger.warning("Found %s in options. This option is predifined and can not be overwritten." , o)
                    self.logger.debug("Removing %s from options" , o)
                    options.remove(d)

        # Find and remove duplicates
        unique_options = {}
        for d in options:
            if d['name'] not in unique_options:
                unique_options[d['name']] = d
            else:
                self.logger.warning("Found duplicate option %s in options. Removing duplicate" , d['name'])

        # Create list of unique options
        options = list(unique_options.values())
        

        # From Candle, can't handle bool, need to fork if we want to support argument groups
        if group:
            group = self.parser.add_argument_group(group)
            print(options)
            print(group)
            parse_from_dictlist( options , group)
        else:
            parse_from_dictlist( options , self.parser)
       

        

    def get_command_line_options(self):
        """Get Command Line Options"""

        self.logger.debug("Getting Command Line Options")
        self.args = self.parser.parse_args()
        self.params = vars(self.args)

    def _check_option(self,option) -> bool:
        pass
    

    # def config(self, section) -> BaseConfig :
    #     cfg=BaseConfig.Config()
    #     if self.params['config_file']:
    #         if os.path.isfile(self.params['config_file']) :
    #             self.logger.info('Loading Config from %s', self.params['config_file'])
    #             cfg.file = self.params['config_file']
    #             cfg.load_config()
    #         else:
    #             self.logger.critical("Can't load Config from %s", self.params['config_file'])
    #     else: 
    #         self.logger.debug('No config file')

    #     # Set params in config
    #     for k in self.params :
    #         (value,error) = cfg.param(section=section, key=k , value=self.params[k])

    #     return cfg


    def initialize_parameters( self,
                              pathToModelDir,
                              default_model=None,
                              additional_definitions=None,
                              required=None,):
        pass


if __name__ == "__main__":
    cli=CLI()
    defaults=[{ 'action' : 'store' , 'choices' : [ 'A' , 'B' , 'C' ] , 'type' : str , 'name' : "dest" }]
    cli.set_command_line_options(options=defaults)
    cli.get_command_line_options()
    # cfg=cli.config("Preprocess")
   
   
    # for k in cli.params :
    #     print("\t".join([k , cli.params[k]]))
    # print(cfg.dict(section="Preprocess"))
    # setattr(cfg, "version" , "0.1")
    # print(cfg.version)
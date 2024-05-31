import logging
import argparse
import os



class CLI:
    """Command Line Options for CSA"""

    def __init__(self):

        # Default format for logging
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)

        # attributes
        self.parser=argparse.ArgumentParser(description='IMPROVE Command Line Parser')
        self.logger=logging.getLogger('CLI')
        self.args = None # Placeholder for args from argparse
        self.params = None # dict of args
        
        # set logger level
        self.logger.setLevel(os.getenv("IMPROVE_LOG_LEVEL" , logging.INFO))
      

        # Set default options
        self.parser.add_argument('-i', '--input_dir', metavar='DIR', type=str, dest="input_dir",
                                  default=os.getenv("IMPROVE_INPUT_DIR" , "./"), 
                                  help='Base directory for input data. Default is IMPROVE_DATA_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        self.parser.add_argument('-o', '--output_dir', metavar='DIR', type=str, dest="output_dir",
                                  default=os.getenv("IMPROVE_OUTPUT_DIR" , "./"), 
                                  help='Base directory for output data. Default is IMPROVE_OUTPUT_DIR or if not specified current working directory. All additional relative output pathes will be placed into the base output directory.')
        self.parser.add_argument('--log_level', metavar='LEVEL', type=str, dest="log_level", 
                                  choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
                                  default=os.getenv("IMPROVE_LOG_LEVEL", "WARNING"), help="Set log levels. Default is WARNING. Levels are:\
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET") 
        self.parser.add_argument('-parsl', '--parsl_config_file', metavar='INI_FILE', dest="parsl_config_file", 
                                  type=str,
                                  default='parsl_config.ini', help="Config file for Parsl in INI format.") 
        self.parser.add_argument('-csa', '--csa_config_file', metavar='INI_FILE', dest="csa_config_file", 
                                  type=str,
                                  default='csa_config.ini', help="Config file for Parsl in INI format.") 
                                  


    def set_command_line_options(self,options=[]):
        """Set Command Line Options, saveguard standard options."""

        self.logger.debug("Setting Command Line Options")
        for o in ['input_dir', 'output_dir', 'log_level', 'config_file']:
            # check if o is the value of name in one of the dicts in options

            if o in options:
                self.logger.warning("Found %s in options. This option is predifined and can not be overwritten." , o)
                self.logger.debug("Removing %s from options" , o)
                del options[o]

            # for d in options:
            #     if isinstance(d, str) :
            #         print(f"Found {d} in options")
            #         if d == o:
            #             self.logger.warning("Found %s in options. This option is predifined and can not be overwritten." , o)
            #             self.logger.debug("Removing %s from options" , o)
            #             options.remove(d)

            #     elif isinstance(d, dict):
            #         if o == d['name']:
            #             self.logger.warning("Found %s in options. This option is predifined and can not be overwritten." , o)
            #             self.logger.debug("Removing %s from options" , o)
            #             options.remove(d)

        #print(options)
        # From Candle, can't handle bool
        for k in range(len(options)):
            self.logger.debug("Adding %s to Command Line Options" , options[k]['name'])
            self.parser.add_argument('--'+options[k]['name'], metavar='VALUE', type=options[k]['type'], dest=options[k]['name'], 
                                     default=options[k]['default'], help=options[k]['help']
                                     #, choices=options[k]['choices'], nargs=options[k]['nargs']
                                     )
       

        

    def get_command_line_options(self):
        """Get Command Line Options"""

        self.logger.debug("Getting Command Line Options")
        self.args = self.parser.parse_args()
        self.params = vars(self.args)
        self.logger.setLevel(self.params['log_level'])
        self.logger.debug("Command Line Options: %s", self.params)


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
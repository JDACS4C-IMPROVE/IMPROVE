import logging
import argparse
import os
from candle import parse_from_dictlist


class CLI:
    """Base Class for Command Line Options"""

    def __init__(self):

        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)
    
        self.parser=argparse.ArgumentParser(description='IMPROVE Command Line Parser')
        self.logger=logging.getLogger('CLI')
        
        # set logger level
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel("DEBUG")

        # Set default options
        self.parser.add_argument('-i', '--input_dir', metavar='DIR', type=dir, dest="input_dir",
                                  default=os.getenv("IMPROVE_INPUT_DIR" , "./"), 
                                  help='Base directory for input data. Default is IMPROVE_DATA_DIR or if not specified current working directory. All additional input pathes will be relative to the base input directory.')
        self.parser.add_argument('-o', '--output_dir', metavar='DIR', type=dir, dest="output_dir",
                                  default=os.getenv("IMPROVE_OUTPUT_DIR" , "./"), 
                                  help='Base directory for output data. Default is IMPROVE_OUTPUT_DIR or if not specified current working directory. All additional relative output pathes will be placed into the base output directory.')
        self.parser.add_argument('--log_level', metavar='LEVEL', type=str, dest="log_level", 
                                  choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
                                  default="WARNING", help="Set log levels. Default is WARNING. Levels are:\
                                      DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET") 
        self.parser.add_argument('-cfg', '--config_file', metavar='INI_FILE', dest="log_level", 
                                  type=argparse.FileType('r'),
                                  default=None, help="Config file in INI format.") 
                                  


    def set_command_line_options(self,options=[]):
        self.logger.debug("Setting Command Line Options")
        parse_from_dictlist( options , self.parser)
        self.parser.add_argument('integers', metavar='N', type=int, nargs='+',
                                 help='an integer for the accumulator')

        

    def get_command_line_options(self):

        # set standard options
        args = self.parser.parse_args()



        pass

    def _check_option(self,option) -> bool:
        pass
    

    def config():
        pass

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
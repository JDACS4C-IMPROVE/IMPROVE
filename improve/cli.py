import logging
import argparse

class CLI:
    """Base Class for Command Line Options"""

    def __init__(self):
        self.parser=argparse.ArgumentParser(description='IMPROVE Command Line Parser')
   
    
    def set_command_line_options(self,options=[]):
        pass

    def get_command_line_options(self):

        # set standard options
        pass

    def _check_option(self,option) -> bool:
        pass
    

    def config():
        pass



if __name__ == "__main__":
    cli=CLI()
    cli.set_command_line_options()
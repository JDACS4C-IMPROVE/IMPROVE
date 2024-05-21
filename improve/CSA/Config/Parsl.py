from parsl.config import Config
from Common import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()

    # Add your custom configuration options here
    # Example:
    # def set_custom_option(self, value):
    #     self.custom_option = value








if __name__ == "__main__":
    cfg=Config()
    defaults=[{ 'action' : 'store' , 'choices' : [ 'A' , 'B' , 'C' ] , 'type' : str , 'name' : "dest" }]
    cli.set_command_line_options(options=defaults)
    cli.get_command_line_options()
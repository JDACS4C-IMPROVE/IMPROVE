from parsl.config import Config
from improve.CSA.Config.Base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self):
        super().__init__()

    # Add your custom configuration options here
    # Example:
    # def set_custom_option(self, value):
    #     self.custom_option = value



    def initialize_parameters(self, cli=None, config_file=None, additional_definitions=None, required=None):
        self.logger.debug(f"Initializing parameters for %s", "PARSL")
        
        return super().initialize_parameters(cli, "PARSL", config_file, additional_definitions, required)
    




if __name__ == "__main__":
    from improve.CSA.CLI import CLI
    cli=CLI()
    cli.set_command_line_options()
    cli.get_command_line_options()
    cfg=Config()
    cfg.logger.setLevel("DEBUG")
    cfg.initialize_parameters(cli=cli)
    # cfg.load_config('parsl.config.ini')
    print(cfg.option)
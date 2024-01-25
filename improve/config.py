import os
import logging
import configparser



class Config:

    ConfigError = str

    def __init__(self) -> None:

        # Default format 
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)
      
       
        self.params = {}
        self.file = None
        self.logger = logging.getLogger('Config')
        self.config = configparser.ConfigParser() 
    
        
    def load_config(self, ):

        if self.file and os.path.isfile(self.file) :
            self.logger.info("Loading config from %s" , self.file)
            self.config.read(self.file)
        else:
            self.logger.error("Can't load config from %s", str(self.file))

        

    def save_config(self,):
        pass

    def param(self, section="DEFAULT" , key=None , value=None) -> list[str, ConfigError]:
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

    def dict(self) -> dict :
        return self.config
        pass

if __name__ == "__main__":
    cfg=Config()
    cfg.file="default.config"
    cfg.load_config()
    print(cfg.params)
    print(cfg.dict())
    print(cfg.param( None , 'weihgts' , None))
    cfg.param('Infer' , 'weihgts' ,'default.weights')
    for section in cfg.config.items():
        print(section)
        for item in cfg.config.items(section[0], raw=False):
            print(item)
    print(cfg.param( "Infer" , 'weihgts' , None))
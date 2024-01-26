import os
import logging
import configparser

from pathlib import Path



class Config:

    # ConfigError = str

    def __init__(self) -> None:

        # Default format 
        FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'
        logging.basicConfig(format=FORMAT)
       
        self.params = {}
        self.file = None
        self.input_dir = None
        self.output_dir = None
        self.logger = logging.getLogger('Config')
        self.logger.setLevel(logging.DEBUG)
        self.config = configparser.ConfigParser()

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

        

    def save_config(self, file , config=None):
        if os.path.isabs(file):
            self.config.read(file)
        else:
            path = Path(config['output_dir'] , file)

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

    def dict(self, section=None) -> dict :

        params = {}
        sections=[]

        if section :
            sections=[section]
        else:
            sections=self.config.sections()
        
        if section:
            for i in self.config.items(section):
                params[i[0]]=i[1]
        else:
            for s in self.config.sections():
                params[s]={}
                for i in self.config.items(s):
                    params[s][i[0]]=i[1]

        return params
    

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
    print(cfg.dict('Infer'))
    cfg.save_config("./tmp/saved.config" , config=cfg.config['DEFAULT'])
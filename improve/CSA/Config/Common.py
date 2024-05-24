import configparser
import json
import logging

FORMAT = '%(levelname)s %(name)s %(asctime)s:\t%(message)s'

class Config():
    def __init__(self):
        self.option = None
    

        logging.basicConfig(format=FORMAT)
        self.logger=logging.getLogger('Common.Config')
        

    def load_config(self, file):
        if file.endswith('.ini'):
            config = configparser.ConfigParser()
            config.read(file)
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

if __name__ == "__main__":
    cfg = Config()
    cfg.logger.setLevel(logging.DEBUG)
    
    cfg.load_config('config.ini')
    cfg.save_config('config.json')
    cfg.save_config('config.ini')
    cfg.save_config('config.txt')
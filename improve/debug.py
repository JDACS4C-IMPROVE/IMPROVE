
class Base():

    def __init__(self) -> None:
        super().__init__()
 
        self.input_dir = None
        self.x_data_path = None
        self.y_data_path = None
        self.splits_path = None

    def set_input_dir(self, input_dir: str) -> None:
        me = self.__class__
        self.__class__=Base
        self.input_dir = self.encode(input_dir)
        self.__class__=me
    
    def encode(self, data: str) -> str:
        code = len(data)
        return code
    
    def wtf(self) -> None:
        print("WTF")

class Debug(Base):

    def __init__(self) -> None:
        super().__init__()
        self.verbose = False

    def set_input_dir(self, input_dir: str) -> None:
        return super().set_input_dir(input_dir)
    
    def encode(self, data: str) -> str:
        return len(data) * 10 
    

if __name__ == "__main__":
    d = Debug()
  
    d.set_input_dir("Hallo")
    print(d.input_dir)
    print(d.encode("Hello"))
  
    b = Base()
    b.set_input_dir("Hello")
    print(b.input_dir)
   


class BaseModel:
    def __init__(self, model_name:str):
        self.model_name = model_name
        pass

    def process_input(self, texts):
        raise NotImplementedError("Subclasses should implement this method")
    
    def generate(self):
        raise NotImplementedError("Subclasses should implement this method")



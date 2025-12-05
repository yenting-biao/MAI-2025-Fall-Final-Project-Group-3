
class BaseModel:
    def __init__(self, model_name:str):
        self.model_name = model_name
        self.messages = None
        pass

    def process_input(self, conversation:list[dict]):
        raise NotImplementedError("Subclasses should implement this method")

    def generate(self) -> str:
        raise NotImplementedError("Subclasses should implement this method")



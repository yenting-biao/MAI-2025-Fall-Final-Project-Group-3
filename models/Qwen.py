from basemodel import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

#   Need to download the ffmpeg package for audio processing in your environment
#   pip uninstall transformers
#   pip install transformers==4.43.0


class Qwen_Audio_Chat(BaseModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="Qwen_Audio_Chat")
        self.device = device
        #   Load actual model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True, cache_dir="./cache")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map=device, trust_remote_code=True, cache_dir="./cache").eval()
        
    
    def process_input(self, conversation:list[dict]) -> None:
        self.messages = self.tokenizer.from_list_format(conversation)
        return 
        # return super().process_input(conversation)
    
    def generate(self) -> str:
        response, history = self.model.chat(
            self.tokenizer,
            query=self.messages,
            history=[],
        )
        return response


if __name__ == "__main__":
    model = Qwen_Audio_Chat(device="cuda" if torch.cuda.is_available() else "cpu")
    print("Qwen Audio Chat model initialized.")
    inputs = [
        {'audio': '../samples/sd-qa_1008642825401516622.wav'}, 
        {'text': 'what does the person say?'},
    ]
    model.process_input(inputs)
    response = model.generate()
    
    print("Model response : ", response)
     
    
    




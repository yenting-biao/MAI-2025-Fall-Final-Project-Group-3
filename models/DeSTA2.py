from basemodel import BaseModel
import torch
from transformers import AutoModel
import os

#   Replace with your Hugging Face token , remember to get the access to META LLAMA3 models
HF_TOKEN = os.environ.get("HF_TOKEN")
#   Transformer version requirement
#   pip uninstall transformers
#   pip install transformers==4.51.3

class DeSTA2(BaseModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="DeSTA2")
        self.device = device
        #   Load actual model 
        self.model = AutoModel.from_pretrained("DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True, cache_dir="./cache", token=HF_TOKEN).to(device)
    
    def process_input(self, conversation):
        self.messages = conversation
        return 
        # return super().process_input(conversation)
    
    def generate(self):
        generate_ids = self.model.chat(
            messages=self.messages,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        response = self.model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        return response
        # return super().generate()


if __name__ == "__main__":
    model = DeSTA2(device="cuda" if torch.cuda.is_available() else "cpu")
    print("DeSTA2 model initialized.")
    messages = [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "audio", "content": "../samples/sd-qa_1008642825401516622.wav"},
            {"role": "user", "content": "What is the content of the audio?"}
        ]
    model.process_input(messages)
    response = model.generate()
    print("Model response:", response)
    
    




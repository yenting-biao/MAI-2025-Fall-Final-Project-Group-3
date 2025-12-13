from models.basemodel import BaseModel
import torch
from transformers import AutoModel
import os
from desta import DeSTA25AudioModel
#   Replace with your Hugging Face token , remember to get the access to META LLAMA3 models
HF_TOKEN = os.environ.get("HF_TOKEN")
#   Transformer version requirement
#   pip uninstall transformers
#   pip install transformers==4.51.3

class DeSTA2_5(BaseModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="DeSTA2_5")
        self.device = device
        #   Load actual model 
        self.model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B", trust_remote_code=True, token=HF_TOKEN, dtype="auto").to(device)
    
    def process_input(self, conversation):
        messages = []
        ICL_examples = len(conversation) - 1
        messages.append({"role": "system", "content": "You are a helpful voice assistant. You will be provided with {} example pairs of questions and answers based on audio inputs. You should follow the examples to answer the last question.".format(ICL_examples)})
        for i in range(ICL_examples):
            info :dict = conversation[i]
            messages.append({"role": "user", 
                             "content": "<|AUDIO|>"+info["instruction"], 
                             "audios": [{
                                 "audio" : info["audio_path"],
                                 "text": None
                             }]})
            messages.append({"role": "assistant", "content": info["answer"]})
        #   Append test example
        messages.append({"role": "user", 
                         "content": "<|AUDIO|>"+conversation[-1]["instruction"], 
                         "audios": [{
                             "audio" : conversation[-1]["audio_path"],
                             "text": None
                         }]})
        self.messages = messages
        return 
        # return super().process_input(conversation)
    
    def generate(self):
        outputs = self.model.generate(
            messages=self.messages,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs.text[0]
        # return super().generate()





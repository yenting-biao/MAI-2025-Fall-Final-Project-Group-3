from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from basemodel import BaseModel
import torch

class Qwen2_Audio_Chat(BaseModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="Qwen2_Audio_Chat")
        self.device = device
        #   Load actual model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", cache_dir="./cache", device_map=device, trust_remote_code=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map=device, cache_dir="./cache", trust_remote_code=True).eval()

    
    def process_input(self, conversation:list[dict]) -> None:
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()), 
                            sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True).to(self.device)
        inputs.input_ids = inputs.input_ids.to(self.device)
        self.inputs = inputs
        return
    
    def generate(self) -> str:
        generate_ids = self.model.generate(**self.inputs, max_length=256)
        generate_ids = generate_ids[:, self.inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response



conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"},
    ]},
    {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"},
    ]},
]

if __name__ == "__main__":
    model = Qwen2_Audio_Chat(device="cuda" if torch.cuda.is_available() else "cpu")
    print("Qwen2 Audio Chat model initialized.")
    model.process_input(conversation)
    response = model.generate()
    
    print("Model response : ", response)
# from .basemodel import BaseModel
from basemodel import BaseModel
from transformers import (
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import librosa
import json

"""
1. Use whisper-large-v3 to convert audio to text
2. Use the corresponding LLM to generate response based on the transcribed text
"""

ValidLLMs = [
    'meta-llama/Llama-3.1-8B-Instruct', 
    'Qwen/Qwen-7B-Chat', 
    'Qwen/Qwen-7B', 
    'Qwen/Qwen2.5-7B',
]

class CascadeModel(BaseModel):
    def __init__(self, llm_model_name: str, device: str = "cuda"):
        assert llm_model_name in ValidLLMs, f"LLM model '{llm_model_name}' is not supported. Choose from {ValidLLMs}."
        self.device = device
        super().__init__(model_name=f"CascadeModel_with_{llm_model_name}")
        # Load audio2text file 
        self.AUDIO2TEXT = json.load(open("../audio_caption/audio_captions.json", "r"))
        
        # Load LLM model for response generation
        self.llm_processor = AutoProcessor.from_pretrained(llm_model_name, trust_remote_code=True)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.llm_model = (
            AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map=device,
                trust_remote_code=True,
                cache_dir="../cache",
            ).eval()
        )

    def _Info2String(self, info: list[dict]) -> str:
        audio_str = ""
        for i in info:
            audio_str += f"[{i['time_segment']}]\n"
            audio_str += f"Text: \"{i['ASR']}\"\n"
            audio_str += f"Gender: {i['Gender']}\n"
            audio_str += f"Emotion: {i['Emotion']}\n"
            audio_str += "\n\n"
        return audio_str
    
    def process_input(self, raw_conversation: list[dict]) -> None:
        # Step 1: Transcribe audio to text using Whisper
        for message in raw_conversation:
            if "audio_path" in message:
                audio_id = message["audio_path"].split("/")[-1]
                audio_input = self.AUDIO2TEXT.get(audio_id, "")
                message["audio_info"] = audio_input
                
        # Step 2: Generate response using LLM based on transcriptions
        ICL_examplenums = len(raw_conversation) - 1
        conversation = []
        conversation.append({"role": "system", "content": "You are a helpful voice assistant. You will be provided with {} example pairs of questions and answers based on audio inputs. You should follow the examples to answer the last question. Only output the final answer in English without any additional words.".format(ICL_examplenums)})
        ICL_examples = raw_conversation[:-1]
        for message in ICL_examples:
            audio_info = message.get("audio_info", "N/A")
            audio_str = self._Info2String(audio_info)
            conversation.append({
                "role": "user", 
                "content": f"Below is a structured analysis of an audio file. \nEach segment contains: a time range, the transcribed speech, the speaker's perceived gender, the detected emotion \n Please use this information to answer the question.\n" + audio_str + f"Question: {message['instruction']}"
            }) 
            conversation.append({
                "role": "assistant", 
                "content": message["ans"]
            })
        
        #   Test example
        message = raw_conversation[-1]
        audio_info = message.get("audio_info", "N/A")
        audio_str = self._Info2String(audio_info)
        conversation.append({"role": "user", "content": f"Below is a structured analysis of an audio file. \nEach segment contains: a time range, the transcribed speech, the speaker's perceived gender, the detected emotion \n Please use this information to answer the question.\n" + audio_str + f"Question: {message['instruction']}"})
        
        self.messages = conversation
        return

    def generate(self) -> str:
        encoding = self.llm_tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(torch.device(self.device))
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        output_ids = self.llm_model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=512,
        ) 
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        output_text = self.llm_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return output_text.strip()
    
Test_message = [{
    "audio_path": "../data/audios/MMAU/516653d5-79d7-404e-a208-62367fdc59b7.wav", 
    "instruction": "Convert the provided spoken statement into text.\nYour entire response should be in all capital letters."
}]

if __name__ == "__main__":
    cascade_model = CascadeModel(llm_model_name='meta-llama/Llama-3.1-8B-Instruct', device="cuda")

    cascade_model.process_input(Test_message)
    response = cascade_model.generate()
    print(response)
    

    
from .basemodel import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import json

"""
1. Load pre-computed audio-to-text captions from ./audio_caption/audio_captions.json
2. Use the corresponding LLM to generate a response based on the transcribed text
"""

ValidLLMs = [
    'meta-llama/Llama-3.1-8B-Instruct', 
    'Qwen/Qwen-7B-Chat', 
    'Qwen/Qwen2.5-7B-Instruct',
]

class CascadeModel(BaseModel):
    def __init__(self, llm_model_name: str, device: str = "cuda"):
        assert llm_model_name in ValidLLMs, f"LLM model '{llm_model_name}' is not supported. Choose from {ValidLLMs}."
        self.device = device
        super().__init__(model_name=f"{llm_model_name}")
        # Load audio2text file 
        with open("./audio_caption/audio_captions.json", "r") as f:
            self.AUDIO2TEXT = json.load(f)
        
        # Load LLM model for response generation
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True, cache_dir="./cache")
        self.llm_model = (
            AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map=device,
                trust_remote_code=True,
                cache_dir="./cache",
            ).eval()
        )

    def _Info2String(self, info: list[dict]) -> str:
        audio_str = ""
        for i in info:
            audio_str += f"{i['time_segment']} {i['ASR']} (Gender: {i['Gender']}, Emotion: {i['Emotion']}).\n"
        return audio_str
    
    def process_input(self, raw_conversation: list[dict]) -> None:
        if (self.model_name in ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']):
            # Step 1: Transcribe audio to text using Whisper
            for message in raw_conversation:
                if "audio_path" not in message:
                    raise ValueError("Each message must contain 'audio_path' key.")
                audio_id = message["audio_path"].split("/")[-1]
                audio_input = self.AUDIO2TEXT.get(audio_id, "")
                message["audio_info"] = audio_input
            
                    
            # Step 2: Generate response using LLM based on transcriptions
            ICL_examplenums = len(raw_conversation) - 1
            conversation = []
            conversation.append({"role": "system", "content": "You are a helpful voice assistant. You will be provided with {} example pairs of questions and answers based on audio inputs. You should follow the examples to answer the last question.".format(ICL_examplenums)})
            ICL_examples = raw_conversation[:-1]
            for message in ICL_examples:
                audio_info = message.get("audio_info", "N/A")
                audio_str = self._Info2String(audio_info)
                conversation.append({
                    "role": "user", 
                    "content": f"Below is an analysis of an audio file. \nEach line contains: a time range, the transcribed speech, the speaker's perceived gender, the detected emotion \n Please use this information to answer the question.\n" + audio_str + f"Question: {message['instruction']}"
                }) 
                conversation.append({
                    "role": "assistant", 
                    "content": message["answer"]
                })
            
            #   Test example
            message = raw_conversation[-1]
            audio_info = message.get("audio_info", "N/A")
            audio_str = self._Info2String(audio_info)
            conversation.append({"role": "user", "content": f"Below is an analysis of an audio file. \nEach line contains: a time range, the transcribed speech, the speaker's perceived gender, the detected emotion \n Please use this information to answer the question. Only answer what is explicitly asked.\n" + audio_str + f"Question: {message['instruction']}"})
            
            self.messages = conversation
        
        elif (self.model_name == 'Qwen/Qwen-7B-Chat'):
            for message in raw_conversation:
                if "audio_path" not in message:
                    raise ValueError("Each message must contain 'audio_path' key.")
                audio_id = message["audio_path"].split("/")[-1]
                audio_input = self.AUDIO2TEXT.get(audio_id, "")
                message["audio_info"] = audio_input
            conversation = "" 
            ICL_examplenums = len(raw_conversation) - 1
            conversation += "You are a helpful voice assistant. You will be provided with {} example pairs of questions and answers based on audio inputs. You should follow the examples to answer the last question.\n".format(ICL_examplenums)
            ICL_examples = raw_conversation[:-1]
            for message in ICL_examples:
                audio_info = message.get("audio_info", "N/A")
                audio_str = self._Info2String(audio_info)
                conversation += f"User: Below is an analysis of an audio file. \nEach line contains: a time range, the transcribed speech, the speaker's perceived gender, the detected emotion \n Please use this information to answer the question.\n" + audio_str + f"Question: {message['instruction']}\n"
                conversation += f"Assistant: {message['answer']}\n"
            
            #   Test example
            message = raw_conversation[-1]
            audio_info = message.get("audio_info", "N/A")
            audio_str = self._Info2String(audio_info)
            conversation += f"User: Below is an analysis of an audio file. \nEach line contains: a time range, the transcribed speech, the speaker's perceived gender, the detected emotion \n Please use this information to answer the question. Only answer what is explicitly asked.\n" + audio_str + f"Question: {message['instruction']}\n"
            
            self.messages = conversation
        
        else:
            raise NotImplementedError(f"Model '{self.model_name}' is not implemented yet.")         ##  This should not happen due to the assertion in __init__
        return

    def _llama_generate_response(self) -> str:
        encoding = self.llm_tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        # Handle both tensor and mapping outputs from apply_chat_template
        if isinstance(encoding, torch.Tensor):
            input_ids = encoding.to(self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            # Assume a dict-like / BatchEncoding object
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)
        output_ids = self.llm_model.generate(
            input_ids=input_ids,        
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=self.llm_tokenizer.eos_token_id,

        ) 
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        output_text = self.llm_tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return output_text.strip()
    
    def _Qwen25_generate_response(self) -> str:
        text = self.llm_tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def _Qwen_generate_response(self) -> str:
        response, history = self.llm_model.chat(self.llm_tokenizer, self.messages, history=None)
        return response.strip()
    
    def generate(self) -> str:
        if self.model_name == 'meta-llama/Llama-3.1-8B-Instruct':
            return self._llama_generate_response()
        elif self.model_name == 'Qwen/Qwen2.5-7B-Instruct':
            return self._Qwen25_generate_response()
        elif self.model_name == 'Qwen/Qwen-7B-Chat':
            return self._Qwen_generate_response()
        else:
            raise NotImplementedError(f"Model '{self.model_name}' is not implemented yet.")         ##  This should not happen due to the assertion in __init__


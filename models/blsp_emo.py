"""
References:
- https://github.com/cwang621/blsp-emo/blob/main/chat_demo.py
- Gemini 3 Pro
"""
import os
from shutil import which

import torch
from transformers import WhisperFeatureExtractor, GenerationConfig
from .blsp_emo_package.src.modeling_blsp2 import Blsp2Model
from .blsp_emo_package.src.tokenization_qwen import QWenTokenizer
from .blsp_emo_package.src.instruction_dataset import get_waveform
from .blsp_emo_package.src.qwen_generation_utils import get_stop_words_ids, decode_tokens

from .basemodel import BaseModel

class ChatHistory(object):
    """Taken from https://github.com/cwang621/blsp-emo/blob/main/chat_demo.py"""
    def __init__(self, 
        tokenizer, 
        extractor, 
        max_window_size=6144,
        max_new_tokens=512,
        use_emotion=False,
        speech_downsample_rate=16
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.max_window_size = max_window_size
        self.max_new_tokens = max_new_tokens
        self.speech_downsample_rate = speech_downsample_rate

        self.im_start_tokens = [tokenizer.im_start_id]
        self.im_end_tokens = [tokenizer.im_end_id]
        self.nl_tokens = tokenizer.encode("\n")

        ### add system
        if use_emotion:
            sys_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            sys_prompt = "You are a helpful assistant."
        input_ids = self.im_start_tokens + self._tokenize_str("system", f"{sys_prompt}") + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.system_histroy = [(input_ids,)]
        self.system_length = input_ids.shape[1]

        self.reset()
    
    def set_system_prompt(self, prompt):
        input_ids = self.im_start_tokens + self._tokenize_str("system", f"{prompt}") + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.system_histroy = [(input_ids,)]
        self.system_length = input_ids.shape[1]
    
    def reset(self):
        self.history = []
        self.lengths = []
        self.cur_length = self.system_length
        self.audio_file = []
        self.audio_to_history = True
    
    def _tokenize_str(self, role, content):
        return self.tokenizer.encode(
            role, allowed_special=set()
        ) + self.nl_tokens + self.tokenizer.encode(content, allowed_special=set())

    def add_text_history(self, role, text):
        input_ids =  self.nl_tokens + self.im_start_tokens + self._tokenize_str(role, text) + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

    def add_audio(self, audio_file):
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech, text=""):
        if self.audio_to_history:
            return
        self.audio_to_history = True
        speech = get_waveform(speech, output_sample_rate=self.extractor.sampling_rate)
        speech_inputs = self.extractor(
            speech,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features.bfloat16()
        speech_attention_mask = speech_inputs.attention_mask

        input_ids = self.nl_tokens + self.im_start_tokens + self._tokenize_str("user", text)
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

        self.history.append(
            (speech_values, speech_attention_mask)
        )
        length = speech_attention_mask.sum().item() // self.speech_downsample_rate
        self.lengths.append(length)
        self.cur_length += length
        

        input_ids = [] + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]
    
    def get_history(self):
        input_ids = self.nl_tokens + self.im_start_tokens + self.tokenizer.encode("assistant")
        input_ids = torch.LongTensor([input_ids])
        length = input_ids.shape[1]

        while self.cur_length > (self.max_window_size - self.max_new_tokens - length):
            pop_length = self.lengths.pop(0)
            self.history.pop(0)
            self.cur_length -= pop_length
        return self.system_histroy + self.history + [(input_ids,)]

    def get_conversation_string(self):
        conversation_parts = []
        
        audio_idx = 1
        for item in self.get_history():
            if len(item) == 1:  # Text tokens
                input_ids = item[0].squeeze(0).tolist()
                decoded = self.tokenizer.decode(input_ids, errors='replace')
                conversation_parts.append(decoded)
            elif len(item) == 2:  # Audio (speech_values, attention_mask)
                conversation_parts.append(f"[AUDIO {audio_idx}]")
                audio_idx += 1
        
        return "".join(conversation_parts)


class BLSP_Emo(BaseModel):
    def __init__(self, model_name: str = "BLSP_Emo", device: str = "cuda", path_to_weights: str = None):
        """
        Initialize the BLSP-Emo model components.
        
        Args:
            model_name (str): Path to the pretrained model directory.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super().__init__(model_name)
        self.device = device

        # Load weights
        if path_to_weights is None:
            path_to_weights = "blsp_emo_weights"
            if not os.path.isdir(path_to_weights):
                if which("hf") is None:
                    raise RuntimeError(
                        "[BLSP-emo] 'hf' CLI tool not found. Please install "
                        "the Hugging Face CLI from https://huggingface.co/docs/huggingface_hub/en/guides/cli."
                    )
                os.system(
                    f"hf download cwang621/blsp-emo --local-dir {path_to_weights}"
                )
        print("[BLSP-emo] Loading model from:", path_to_weights)
        
        # Load Tokenizer
        self.tokenizer = QWenTokenizer.from_pretrained(
            path_to_weights,
        )
        
        # Load Feature Extractor (for Audio)
        self.extractor = WhisperFeatureExtractor.from_pretrained(path_to_weights)
        
        # Load Model
        self.model = Blsp2Model.from_pretrained(
            path_to_weights, 
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        # self.model = self.model.half()
        self.model = self.model.bfloat16()  # don't use self.model.half() as it will cause an error; see https://github.com/meta-llama/llama/issues/380#issuecomment-1656714118
        self.model.eval()
        
        # Load Generation Config
        self.generation_config = GenerationConfig.from_pretrained(path_to_weights)
        
        # Setup Stop Words
        self.stop_words_ids = get_stop_words_ids(
            self.generation_config.chat_format, 
            self.tokenizer
        )

        # Define special tokens used for constructing prompt format
        self.im_start_tokens = [self.tokenizer.im_start_id]
        self.im_end_tokens = [self.tokenizer.im_end_id]
        self.nl_tokens = self.tokenizer.encode("\n")

        # Modify config
        self.generation_config.update(
            **{
                # original generation_config.update from chat_demo.py: (may need to tweak these settings if current window size is too small or response is cut off)
                # "max_new_tokens": 512,
                # "min_new_tokens": 1,
                # "temperature": 0.5,
                # "max_window_size": 6144,
                # "bos_token_id": self.tokenizer.encode("\n")[0],
                # "num_return_sequences": 1,

                "max_new_tokens": 128,
                "min_new_tokens": 1,
                "do_sample": False,  # Greedy decoding
                "temperature": 1.0,
                # Note: it looks like top_p=0.5 and top_k=0 were set in the original GenerationConfig. If we set do_sample=False, we get a warning about do_sample=False and top_p/top_k settings both being set, even if we set top_p and top_k to None. Setting top_p=1.0 silences the warning for top_p but the warning for top_k can't seem to be silenced.
                "top_p": 1.0,
                "top_k": 1,
                "num_beams": 1,
                "num_return_sequences": 1,
                "bos_token_id": self.nl_tokens[0],
            }
        )

        self.history = ChatHistory(self.tokenizer, self.extractor, self.generation_config.max_window_size, self.generation_config.max_new_tokens, False)

    def process_input(self, conversation: list[dict]):
        """
        Process the raw conversation list into the format expected by Blsp2Model.chat().
        Each dict contains: {"audio_path": ..., "instruction": ..., "answer": ...}
        The order is [System] -> [User Text] -> [User Audio] -> [Assistant Text] -> ...
        [User Text] and [User Audio] are delimited by "\n\nSpeech: ".
        Note it appears from small tests that putting [User Audio] before [User Text]
        causes model to either say it did not receive an audio input or treat 
        the audio as part of the text instruction.
        """
        self.history.reset()
        
        for turn in conversation:
            audio_path = turn.get("audio_path")
            instruction = turn.get("instruction", "")
            answer = turn.get("answer")

            if audio_path:
                content = instruction
                if content:
                    content += "\n\nSpeech: "
                self.history.add_audio(audio_path)
                self.history.add_speech_history(self.history.audio_file[-1], text=content)
            elif instruction:
                self.history.add_text_history("user", instruction)
            else:
                raise ValueError("User turn must have either 'instruction' or 'audio_path'.")
            
            if answer is not None:  # means there is an assistant response
                self.history.add_text_history("assistant", answer)
        
        self.messages = self.history.get_conversation_string()

    def generate(self) -> str:
        if not self.history.history:
            raise ValueError("No input processed. Call process_input or pass conversation to generate.")

        # Run the model
        # The model expects the history list-of-tuples we built in process_input
        output = self.model.chat(
            history=self.history.get_history(),
            generation_config=self.generation_config,
            stop_words_ids=self.stop_words_ids,
            device=self.device
        )

        # Decode result
        response = decode_tokens(
            output[0],
            self.tokenizer,
            raw_text_len=0,
            context_length=0,
            chat_format=self.generation_config.chat_format,
            verbose=False,
            errors='replace'
        )

        self.history.reset()

        return response

    def _tokenize_str(self, role, content):
        """Helper from chat_demo.py to encode role + newline + content"""
        return (
            self.tokenizer.encode(role, allowed_special=set()) + 
            self.nl_tokens + 
            self.tokenizer.encode(content, allowed_special=set())
        )
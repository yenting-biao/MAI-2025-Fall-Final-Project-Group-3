"""
References:
- https://github.com/cwang621/blsp-emo/blob/main/generate.py
- Gemini 2.5 Pro
"""

import os
import warnings
from typing import List, Dict, Optional
from shutil import which

import numpy as np
import librosa
import torch
from transformers import WhisperFeatureExtractor, GenerationConfig

from .basemodel import BaseModel
from .blsp_emo_package.src.modeling_blsp2 import Blsp2Model
from .blsp_emo_package.src.tokenization_qwen import QWenTokenizer
from .blsp_emo_package.src.qwen_generation_utils import (
    decode_tokens,
    get_stop_words_ids,
)


class BLSP_emo(BaseModel):
    """
    Wrapper for BLSP-emo (Blsp2Model).

    This class adapts the BLSP-emo model to the BaseModel interface,
    handling audio feature extraction, QWen tokenization, and generation.

    Expected conversation format (minimal):
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "question", "audio_path": "foo.wav"},
            ...
        ]

    Like the DiVA example, this implementation:
      - Uses ONLY the latest user message.
      - If `audio_path` (or `audio` as np.ndarray) exists, it's processed.
      - If not, it falls back to text-only.
    """

    def __init__(
        self,
        model_name: str = "cwang621/blsp-emo",
        device: Optional[str] = None,
        sampling_rate: int = 16_000,
        path_to_weights: Optional[str] = None,
    ) -> None:
        """Initialize the BLSP-emo wrapper.

        Args:
            model_name: Name of model.
            device: Overrides automatic device selection ("cuda" or "cpu").
            sampling_rate: The sampling rate for audio processing.
            path_to_weights: Path to the folder containing the files downloaded from https://huggingface.co/cwang621/blsp-emo. If None, this function either assumes the weights are stored in the folder `blsp_emo_weights`, or, if not found, will run `hf download cwang621/blsp-emo --local-dir {path_to_weights}` to download the weights to a local folder named `blsp_emo_weights`.
        """
        super().__init__(model_name)

        # 1. Set up device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn(
                    "[BLSP-emo] CUDA is not available; running on CPU. "
                    "Inference will be very slow.",
                    RuntimeWarning,
                )
        else:
            self.device = device
            if self.device == "cpu":
                warnings.warn(
                    "[BLSP-emo] Model explicitly set to run on CPU. "
                    "Inference will be very slow.",
                    RuntimeWarning,
                )

        self.sampling_rate = sampling_rate
        load_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 2. Load model, tokenizer, and extractor
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
        self.tokenizer = QWenTokenizer.from_pretrained(path_to_weights)
        self.extractor = WhisperFeatureExtractor.from_pretrained(path_to_weights)
        self.model = Blsp2Model.from_pretrained(
            path_to_weights, torch_dtype=load_dtype
        ).to(
            self.device
        )  # this takes about 8 minutes on 3090 GPU

        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.dtype = next(
            self.model.parameters()
        ).dtype  # ensure input dtype matches model, otherwise may error at models/blsp_emo_package/src/plora.py line 721 `result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)` "RuntimeError: mat1 and mat2 must have the same dtype"
        print(f"[BLSP-emo] Model loaded with actual dtype: {self.dtype}")

        # 3. Load generation configuration
        self.generation_config = GenerationConfig.from_pretrained(path_to_weights)

        # Get special prompt tokens
        self.nl_tokens = self.tokenizer.encode("\n")

        # Apply default generation settings from the script
        self.generation_config.update(
            **{
                "max_new_tokens": 128,
                "min_new_tokens": 1,
                "do_sample": False,  # Greedy decoding
                "temperature": 1.0,
                "top_p": None,
                "top_k": None,
                # Note: it looks like top_p=0.5 and top_k=0 were set in the original GenerationConfig,
                # so if we set do_sample=False here we get a warning. Maybe removing
                # do_sample=False here and just setting top_p=None and top_k=None 
                # would silence this warning. This is left as future work.
                "num_beams": 1,
                "num_return_sequences": 1,
                "bos_token_id": self.nl_tokens[0],
            }
        )

        self.stop_words_ids = get_stop_words_ids(
            self.generation_config.chat_format, self.tokenizer
        )

        # 4. Store prompt tokens
        self.im_start_tokens = [self.tokenizer.im_start_id]
        self.im_end_tokens = [self.tokenizer.im_end_id]

        # 5. Internal state variables to pass data between methods
        self._input_ids: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._suffix_input_ids: Optional[torch.Tensor] = None
        self._suffix_attention_mask: Optional[torch.Tensor] = None
        self._speech_values: Optional[torch.Tensor] = None
        self._speech_attention_mask: Optional[torch.Tensor] = None

    def _get_last_user_msg(self, conversation: List[Dict]) -> Dict:
        """Finds the last message from the 'user' in the conversation.

        For example, given:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What does the speaker say in the audio?",
                "audio_path": "samples/sd-qa_1008642825401516622.wav",
            },
        ],
        this function returns conversation[1].
        """
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                return msg
        raise ValueError("Conversation has no user message.")

    def _load_audio(self, path: str) -> np.ndarray:
        """Loads and resamples audio from a file path."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
        return audio

    def _tokenize_str(self, role: str, content: str) -> List[int]:
        """Helper to tokenize role and content. This function is taken directly from https://github.com/cwang621/blsp-emo/blob/main/generate.py."""
        return (
            self.tokenizer.encode(role, allowed_special=set())
            + self.nl_tokens
            + self.tokenizer.encode(content, allowed_special=set())
        )

    def process_input(self, conversation: List[Dict], use_emotion: bool = False) -> None:
        """Prepare BLSP-emo inputs from a conversation.

        This function tokenizes the text and extracts audio features,
        storing them in internal state variables for `generate()`.

        Currently assumes there is only one user message in the conversation (i.e., single turn).
        """
        # 1. Get the last user message
        user_msg = self._get_last_user_msg(conversation)

        # 2. Get text instruction (content)
        instruction = user_msg.get("content", "") or ""

        # 3. Load audio (if present)
        speech_array: Optional[np.ndarray] = None
        if "audio" in user_msg and isinstance(user_msg["audio"], np.ndarray):
            speech_array = user_msg["audio"]
        elif "audio_path" in user_msg:
            try:
                speech_array = self._load_audio(user_msg["audio_path"])
            except Exception as e:
                warnings.warn(f"Could not load audio file: {e}", RuntimeWarning)
                speech_array = None
        if len(instruction.strip()) > 0 and speech_array is not None:
            # Following the example in https://github.com/cwang621/blsp-emo/blob/main/README.md,
            # we separate user text instruction and audio with "\n\nSpeech: ", otherwise the model
            # may think they are part of the same instruction, sometimes even
            # getting confused and responding with something along the lines of
            # "I do not have access to audio content".
            # See blsp_sample_outputs.py for examples of this behavior.
            instruction += "\n\nSpeech: "

        # 4. Check that user input is not empty
        if speech_array is None and not instruction:
            raise ValueError(
                "User message has neither audio nor text content; "
                "nothing to send to BLSP-emo."
            )

        # 5. Process text inputs (build prompt as per script)

        # 5.1. Set system prompt
        if use_emotion:
            system_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            system_prompt = "You are a helpful assistant."

        # 5.2. Build input prompt
        input_ids_list = (
            self.im_start_tokens
            + self._tokenize_str("system", system_prompt)
            + self.im_end_tokens
            + self.nl_tokens
            + self.im_start_tokens
            + self._tokenize_str("user", instruction)
        )

        # 5.3. Build suffix prompt
        suffix_input_ids_list = (
            self.im_end_tokens
            + self.nl_tokens
            + self.im_start_tokens
            + self.tokenizer.encode("assistant")
        )

        # Store as tensors (batch size 1)
        self._input_ids = torch.LongTensor([input_ids_list]).to(self.device)
        self._attention_mask = torch.LongTensor([[1] * len(input_ids_list)]).to(
            self.device
        )
        self._suffix_input_ids = torch.LongTensor([suffix_input_ids_list]).to(
            self.device
        )
        self._suffix_attention_mask = torch.LongTensor(
            [[1] * len(suffix_input_ids_list)]
        ).to(self.device)

        # 6. Process audio input
        if speech_array is not None:
            # Use the extractor to get features
            speech_inputs = self.extractor(
                [speech_array],  # Extractor expects a list of arrays
                sampling_rate=self.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )
            self._speech_values = speech_inputs.input_features.to(self.dtype).to(
                self.device
            )
            self._speech_attention_mask = speech_inputs.attention_mask.to(self.device)
        else:
            self._speech_values = None
            self._speech_attention_mask = None

    def generate(self) -> str:
        """Run BLSP-emo generation and return a single string response."""
        # 1. Check if process_input() has been called
        if self._input_ids is None or self._suffix_input_ids is None:
            raise RuntimeError("process_input() must be called before generate().")

        # 2. Run model.generate() with all processed inputs
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=self._input_ids,
                attention_mask=self._attention_mask,
                suffix_input_ids=self._suffix_input_ids,
                suffix_attention_mask=self._suffix_attention_mask,
                speech_values=self._speech_values,
                speech_attention_mask=self._speech_attention_mask,
                generation_config=self.generation_config,
                stop_words_ids=self.stop_words_ids,
            )

        # 3. Decode the output tokens to a string
        # We assume batch size = 1
        output_text = decode_tokens(
            outputs[0],
            self.tokenizer,
            raw_text_len=0,
            context_length=0,
            chat_format=self.generation_config.chat_format,
            verbose=False,
            errors="replace",
        )

        # 4. Clean up state for next run
        self._input_ids = None
        self._attention_mask = None
        self._suffix_input_ids = None
        self._suffix_attention_mask = None
        self._speech_values = None
        self._speech_attention_mask = None

        return output_text

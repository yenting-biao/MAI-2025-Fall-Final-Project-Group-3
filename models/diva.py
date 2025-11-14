import os, warnings
from typing import List, Dict, Optional

import numpy as np
import librosa
import torch
from transformers import AutoModel

from models.basemodel import BaseModel

class DiVA(BaseModel):
    """
    Wrapper for DiVA Llama 3 (speech-in / text-in, text-out).

    Expected conversation format (minimal):
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "question", "audio_path": "foo.wav"},
            ...
        ]

    For now (13 Nov 2025), we:
      - Use ONLY the latest user message.
      - If `audio_path` exists, we load audio and send it to DiVA.
      - If not, we fall back to text-only (best-effort).
    """

    def __init__(
        self,
        model_name: str = "WillHeld/DiVA-llama-3-v0-8b",
        device: Optional[str] = None,
        sampling_rate: int = 16_000,
    ):
        super().__init__(model_name)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                warnings.warn(
                    "[DiVA] CUDA is not available; running on CPU. "
                    "Inference will be very slow.",
                    RuntimeWarning,
                )
        else:
            self.device = device
            if self.device == "cpu":
                warnings.warn(
                    "[DiVA] DiVA has been explicitly set to run on CPU. "
                    "Inference will be very slow.",
                    RuntimeWarning,
                )
        self.sampling_rate = sampling_rate

        # DiVA exposes custom generate() via remote code
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            # dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)

        # will store processed inputs between process_input() and generate()
        self._speech_batch: Optional[List[Optional[np.ndarray]]] = None
        self._style_prompts: Optional[List[str]] = None

    def _get_last_user_msg(self, conversation: List[Dict]) -> Dict:
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                return msg
        raise ValueError("Conversation has no user message.")

    def _load_audio(self, path: str) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, sr = librosa.load(path, sr=self.sampling_rate, mono=True)
        return audio

    def process_input(self, conversation: List[Dict]):
        """
        Prepare DiVA inputs from a conversation.

        - speech_data: list[np.ndarray] or None
        - style prompt: we currently just pass the (text) content as a style / text prompt.
        """
        user_msg = self._get_last_user_msg(conversation)

        # Style / text prompt (can be improved later to include full history)
        style_prompt = user_msg.get("content", "") or ""

        # Optional audio
        speech_array: Optional[np.ndarray] = None
        if "audio" in user_msg and isinstance(user_msg["audio"], np.ndarray):
            speech_array = user_msg["audio"]
        elif "audio_path" in user_msg:
            speech_array = self._load_audio(user_msg["audio_path"])

        # For basic "can it run?" tests, just require any input
        if speech_array is None and not style_prompt:
            raise ValueError(
                "User message has neither audio nor text content; nothing to send to DiVA."
            )

        # DiVA expects a list of examples
        self._speech_batch = [speech_array] if speech_array is not None else [None]
        self._style_prompts = [style_prompt] if style_prompt else [""]

    def generate(self) -> str:
        """
        Run DiVA and return a single string response.

        DiVA's custom generate() API (from its model card example): :contentReference[oaicite:1]{index=1}
           model.generate([speech_data])
           model.generate([speech_data], ["Reply Briefly Like A Pirate"])
        """
        if self._speech_batch is None or self._style_prompts is None:
            raise RuntimeError("process_input() must be called before generate().")

        with torch.no_grad():
            if any(s is not None for s in self._speech_batch):
                outputs = self.model.generate(self._speech_batch, self._style_prompts)
            else:
                # fallback if only text is available; may need to adapt
                outputs = self.model.generate(
                    self._speech_batch, self._style_prompts
                )

        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return str(outputs[0])

        return str(outputs)

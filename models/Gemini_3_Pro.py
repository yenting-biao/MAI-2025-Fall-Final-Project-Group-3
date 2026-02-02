"""Gemini Large Audio-Language Model model wrapper."""

import time
import os
from typing import TypedDict, NotRequired, Any
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

from models.basemodel import BaseModel


class _Message(TypedDict):
    """
    Type definition for a single message in a conversation.

    Attributes:
        audio_path: Path to the audio file for this message.
        instruction: Text instruction or question about the audio.
        answer: Ground-truth answer for in-context learning examples. This is
            None if the message is the test question (the last message in the
            conversation).
    """

    audio_path: str
    instruction: str
    answer: NotRequired[str]


def _exists(obj: Any) -> bool:
    return obj is not None


class Gemini3Pro(BaseModel):
    """Gemini 3 Pro model wrapper with audio and text support."""

    default_generation_config = {
        # Note: setting max_output_tokens will cause response to be empty
        # if the thinking tokens exceed max_output_tokens (i.e., thinking process
        # doesn't end by max_output_tokens). Appears to be a known open issue:
        # https://github.com/googleapis/python-genai/issues/626
        "temperature": 1.0,
        "thinking_config": types.ThinkingConfig(thinking_level="low"),
        # seed is set in run.py
    }

    def __init__(
        self,
        model_name: str = "gemini-3-pro-preview",
        max_retries: int = 5,
        generation_config: dict[str, Any] | None = None,
    ):
        """Initialize the Gemini model wrapper.

        Requires GEMINI_API_KEYS environment variable to be set with
        comma-separated API keys. You can set it in a .env file or set the
        environment variable directly before calling this code.

        Args:
            model_name: Name of the Gemini model to use.
            max_retries: Maximum number of retry attempts on failure. When a Gemini
                API call fails, the wrapper will rotate to the next API key and retry
                until max_retries is reached.
            generation_config: Gemini generation configuration.

        Raises:
            ValueError: If GEMINI_API_KEYS environment variable is not set.
        """

        super().__init__(model_name=model_name)

        load_dotenv()
        api_keys = os.environ.get("GEMINI_API_KEYS", "")
        if not api_keys:
            raise ValueError("GEMINI_API_KEYS environment variable is not set.")
        self.api_keys = [key for key in api_keys.replace(" ", "").split(",") if key]
        if not self.api_keys:
            raise ValueError(
                "GEMINI_API_KEYS environment variable was not properly set."
                ' Format should be "<key1>,<key2>,...,<keyN>" with no trailing commas'
            )

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        self.current_key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        self.model_name = model_name
        self.max_retries = max_retries
        self.contents: list[types.Content] = []
        self.files: list[types.File] = []  # To keep track of uploaded files so
                                           # that we can delete them later.
        self.generation_config = (
            generation_config
            if generation_config is not None
            else self.default_generation_config.copy()
        )

        if max_retries < len(self.api_keys):
            print(
                f"Warning: max_retries ({max_retries}) is less than the number"
                f" of API keys ({len(self.api_keys)})."
            )

        print(f"Initialized Gemini model: {model_name}")
        print(f"Using {len(self.api_keys)} API keys")
        print(f"Using generation configuration: {self.generation_config}")

    def process_input(self, conversation: list[_Message]) -> None:
        """Process a conversation consisting of audio files and instructions.

        Converts the conversation into the format required by the Gemini API,
        uploading audio files and creating user/model turn pairs for in-context
        learning examples.

        Args:
            conversation: Let N = len(conversation). The first N-1 messages are
                in-context learning examples, each containing an audio file path,
                an instruction, and a ground-truth answer. The last message is the
                test question, containing an audio file path and an instruction
                without an answer. Each in-context learning is split into a user
                turn (audio + instruction) and a model turn (answer). The test
                question is given as a user turn only, prompting the model to
                generate the answer. Only one conversation is allowed; batch 
                processing is not supported.

        Raises:
            ValueError: If an audio file specified in the conversation doesn't exist.
        """

        self.contents.clear()
        for file in self.files:
            self.client.files.delete(name=file.name)
        self.files.clear()

        for idx, message in enumerate(conversation):
            audio_path = message["audio_path"]
            instruction = message["instruction"]
            answer: str | None = message.get("answer", None)

            if not Path(audio_path).exists():
                raise ValueError(f"Audio file not found, got {audio_path}")

            # Create user turn consisting of audio and instruction
            user_parts = []
            file = self.client.files.upload(file=audio_path)
            user_parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))
            user_parts.append(types.Part(text=instruction))
            self.contents.append(types.Content(role="user", parts=user_parts))

            # Create assistant response if message is an in-context learning example
            is_icl_example = _exists(answer)
            if is_icl_example:
                assistant_parts = []
                assistant_parts.append(types.Part(text=answer))
                self.contents.append(types.Content(role="model", parts=assistant_parts))
            else:
                assert idx == len(conversation) - 1, (
                    "Only the last message in the conversation can be without an answer."
                )

        assert len(self.contents) == 2 * len(conversation) - 1, (
            f"Expected {2 * len(conversation) - 1} contents, got {len(self.contents)}."
        )

        # Used for debugging/logging purposes by run.py.
        # Each item in self.contents is of type <google.genai.types.Content> and 
        # cannot be serialized by the json module, so we convert self.contents
        # to string here.
        self.messages = str(self.contents)

    def generate(self) -> str:
        """Generate a response from the model based on processed input.

        Please call process_input() before calling this method.

        Returns:
            Generated text response from the model. Empty string if no response.

        Raises:
            ValueError: If process_input() has not been called before generate().
            RuntimeError: If all retry attempts fail.
        """

        if not self.contents:
            raise ValueError("Please call process_input() before calling generate().")

        num_tries = 0
        response = None
        while response is None and num_tries < self.max_retries:
            num_tries += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self.contents,
                    config=types.GenerateContentConfig(**self.generation_config),
                )
            except Exception as e:
                next_key_index = (self.current_key_index + 1) % len(self.api_keys)
                print(
                    f"Error with API key number {self.current_key_index}: {e}"
                    + (
                        f". Trying API key number {next_key_index} next."
                        if num_tries < self.max_retries
                        else ""
                    )
                )
                self.current_key_index = next_key_index
                self.client = genai.Client(
                    api_key=self.api_keys[self.current_key_index]
                )
        if num_tries == self.max_retries and response is None:
            raise RuntimeError(
                f"Gemini failed to generate response in {self.max_retries} tries"
            )

        # Five-second buffer between generations to prevent rate limit error
        time.sleep(5)

        self.contents.clear()  # This is to ensure we don't encounter a silent
                               # bug where old contents are reused when new ones
                               # should be used.
        for file in self.files:
            self.client.files.delete(name=file.name)
        self.files.clear()

        if not response.text:
            print("Warning: Response is empty.")
            return ""
        return response.text

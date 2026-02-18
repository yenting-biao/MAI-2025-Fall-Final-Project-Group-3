"""Gemini Large Audio-Language Model model wrapper."""

import atexit
import time
import os
from typing import TypedDict, NotRequired, Any
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

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


class Gemini(BaseModel):
    """Gemini model wrapper with audio and text support."""

    default_generation_config = {
        # Note: setting max_output_tokens will cause response to be empty
        # if the thinking tokens exceed max_output_tokens (i.e., thinking process
        # doesn't end by max_output_tokens). Appears to be a known open issue:
        # https://github.com/googleapis/python-genai/issues/626
        "temperature": 1.0,  # Temperature officially recommended by Google

        # The default thinking level is usually dynamic if thinking level is not 
        # specified. Default thinking level depends on model, so please check
        # https://ai.google.dev/gemini-api/docs/thinking.

        # Get thinking summary since Gemini does most of its reasoning behind-
        # the-scenes.
        "thinking_config": types.ThinkingConfig(include_thoughts=True),

        # seed is set in run.py
    }

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 5,
        max_upload_retries: int = 3,
        generation_config: dict[str, Any] | None = None,
    ):
        """Initialize the Gemini model wrapper.

        Requires GEMINI_API_KEY environment variable to be set. You can set it 
        in a .env file or set the environment variable directly before calling 
        this code.

        Args:
            model_name: Name of the Gemini model to use. model_name will be passed
                directly to the Gemini API, so it must be a valid model code. See
                https://ai.google.dev/gemini-api/docs/models for valid model codes.
            max_retries: Maximum number of retry attempts on failure. When a Gemini
                API call fails, the wrapper will rotate to the next API key and retry
                until max_retries is reached. If only one API key is provided, 
                the same key will be retried up to max_retries times. Must be at 
                least 1.
            max_upload_retries: Maximum number of retry attempts for file uploads
                in the case of errors like "400 Bad Request: Upload has already
                been terminated.". Must be at least 1.
            generation_config: Gemini generation configuration.

        Raises:
            ValueError: If GEMINI_API_KEY environment variable is not set.
        """

        super().__init__(model_name=model_name)

        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.api_key = api_key

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if max_upload_retries < 1:
            raise ValueError("max_upload_retries must be at least 1.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.max_upload_retries = max_upload_retries
        self.contents: list[types.Content] = []
        self.generation_config = (
            generation_config
            if generation_config is not None
            else self.default_generation_config.copy()
        )

        # Keep track of uploaded files. We only upload each file once and reuse
        # the uploaded file URIs in subsequent requests instead of reuploading
        # all files at every call to process_input(). All uploaded files are deleted
        # at program exit.
        self.uploaded_files: dict[str, types.File] = {}
        atexit.register(self._cleanup_files)

        print(f"Initialized Gemini model: {model_name}")
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
        if hasattr(self, "thinking_summary"):
            del self.thinking_summary  # Clear any previous thinking summary to catch
                                       # bugs where old thinking summaries are reused.

        num_examples = len(conversation) - 1
        self.generation_config["system_instruction"] = f"You are a helpful assistant. You will be provided with {num_examples} example pairs of questions and answers. You should follow the examples to answer the last question."

        for idx, message in enumerate(conversation):
            audio_path = message["audio_path"]
            instruction = message["instruction"]
            answer: str | None = message.get("answer", None)

            # Create user turn consisting of audio (if provided) and instruction
            user_parts = []
            
            if audio_path is not None:
                if not Path(audio_path).exists():
                    raise ValueError(f"Audio file not found, got {audio_path}")

                num_retries = 0
                while num_retries < self.max_upload_retries:
                    num_retries += 1
                    try:
                        if audio_path in self.uploaded_files:
                            file = self.uploaded_files[audio_path]
                        else:
                            file = self.client.files.upload(file=audio_path)
                            self.uploaded_files[audio_path] = file
                        break
                    except ClientError as e:
                        print(f"Error uploading file {audio_path}: {e}")
                        if num_retries == self.max_upload_retries:
                            print("Conversation in question:", conversation)
                            raise RuntimeError(
                                f"Failed to upload file {audio_path} after"
                                f" {self.max_upload_retries} attempts."
                            )
                        else:
                            print("Retrying upload...")
                            time.sleep(1)  # Brief pause before retrying
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
        self.messages = self.generation_config["system_instruction"] + str(self.contents)

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
                print(
                    f"Error with API: {e}"
                    + (
                        f". Retrying..."
                        if num_tries < self.max_retries
                        else ""
                    )
                )
                if num_tries < self.max_retries:
                    time.sleep(1)  # Brief pause before retrying
        if num_tries == self.max_retries and response is None:
            raise RuntimeError(
                f"Gemini failed to generate response in {self.max_retries} tries"
            )

        self.contents.clear()  # This is to ensure we don't encounter a silent
                               # bug where old contents are reused when new ones
                               # should be used.

        if not response.candidates[0].content.parts or len(response.candidates[0].content.parts) < 2 or not response.candidates[0].content.parts[0].thought:
            print("Warning: Model did not think.")
            self.thinking_summary = ""
            self.answer = response.text
        else:
            self.thinking_summary = response.candidates[0].content.parts[0].text
            self.answer = response.candidates[0].content.parts[1].text

        if not self.answer:
            print("Warning: Response is empty.")
            return ""
        return self.answer

    def _cleanup_files(self):
        """Clean up all uploaded files.

        If files are not explicitly deleted, they will take up space in quota
        for 48 hours.

        This function is called at program exit.
        """

        print(f"Cleaning up {len(self.uploaded_files)} uploaded files...")

        for audio_path, file in tqdm(self.uploaded_files.items(), desc="Deleting audio files from Google servers"):
            try:
                self.client.files.delete(name=file.name)
            except Exception as e:
                print(f"Warning: Could not delete file {audio_path}: {e}")
        self.uploaded_files.clear()

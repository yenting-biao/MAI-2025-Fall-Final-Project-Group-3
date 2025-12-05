from typing import List
from google import genai
from google.genai import types
import time
import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

# TODO!!
PROMPT_MAP = {
    "Chain-of-Thought": """
Given the audio file, answer the following question. The answer is "{answer}" However, do not include answer in your reasoning process:

{instruction}
    """,
    "Creative_Writing": {
        "detectable_format:number_bullet_lists": """{instruction}""",
        "keywords:existence": """{instruction}""",
        "keywords:forbidden_words": """""",
        "length_constraints:number_words": """""",
        "length_constraints:number_sentences": """""",
        "length_constraints:number_paragraphs": """""",
    },
}


class Gemini:
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_keys: str = None,
        max_retries: int = 5,
    ):
        if not api_keys:
            raise ValueError("GEMINI_API_KEYS environment variable is not set.")
        self.api_keys = api_keys.split(
            ","
        )  # you can use multiple API keys to avoid rate limit
        self.current_key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        self.model_name = model_name
        self.max_retries = max_retries
        print(f"Initialized Gemini model: {model_name}")

    def generate(
        self,
        audio: str,
        user_prompt: str = None,
        # target_response: str = None,
        # do_sample: bool = False,
        # temperature: float = 1.0,
        # top_p: float = 0.9,
        # max_new_tokens: int = 512,
    ) -> List[str]:
        if user_prompt is None:
            user_prompt = ""

        for _ in range(self.max_retries):
            try:
                if audio:
                    file = self.client.files.upload(file=audio)
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[user_prompt, file],
                        config=types.GenerateContentConfig(
                            # temperature=temperature,
                            # max_output_tokens=max_new_tokens,
                            # top_p=top_p,
                            seed=0,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=-1
                            ),  # dynamic thinking budget for audio input
                        ),
                    )
                else:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[user_prompt],
                        config=types.GenerateContentConfig(
                            # temperature=temperature,
                            # max_output_tokens=max_new_tokens,
                            # top_p=top_p,
                            seed=0,
                            thinking_config=types.ThinkingConfig(thinking_budget=1),
                        ),
                    )

                break  # If successful, exit the loop
            except Exception as e:
                print(f"Error with API key index {self.current_key_index}: {e}")
                self.current_key_index = (self.current_key_index + 1) % len(
                    self.api_keys
                )
                self.client = genai.Client(
                    api_key=self.api_keys[self.current_key_index]
                )

        # wait for 5 seconds to prevent rate limit error
        time.sleep(5)

        return response.text if response.text else ""


def load_metadata_maps(path: str) -> dict:
    metadata_df = pd.read_csv(path)
    transcription_map = dict(zip(metadata_df["filename"], metadata_df["transcription"]))
    emotion_map = dict(zip(metadata_df["filename"], metadata_df["emotion"]))
    gender_map = dict(zip(metadata_df["filename"], metadata_df["gender"]))
    return {
        "transcription": transcription_map,
        "emotion": emotion_map,
        "gender": gender_map,
    }


def main(args):
    metadata = load_metadata_maps(args.metadata_file)

    # Load generated examples
    with open(args.input_file, "r") as f:
        example_data = json.load(f)

    if args.api_keys:
        gemini_model = Gemini(api_keys=args.api_keys)
    else:
        gemini_model = Gemini()

    if args.constraint_type == "chain-of-thought":
        prompt_template = PROMPT_MAP["Chain-of-Thought"]
    else:
        prompt_template = PROMPT_MAP["Creative_Writing"][args.subconstraint]

    task_data = example_data[args.task][args.constraint_type]
    if args.constraint_type == "creative_writing":
        task_data = task_data[args.subconstraint]

    for item in tqdm(task_data[: args.test_num]):
        audio_path = "./audios/" + item["audio_path"]
        instruction = item["instruction"]

        transcription = metadata["transcription"][audio_path.split("/")[-1]]
        emotion = metadata["emotion"][audio_path.split("/")[-1]]
        gender = metadata["gender"][audio_path.split("/")[-1]]

        if args.task == "ASR":
            answer = transcription
        elif args.task == "SER":
            answer = emotion
        elif args.task == "GR":
            answer = gender

        full_prompt = prompt_template.replace("{instruction}", instruction).replace(
            "{answer}", answer
        )

        response = gemini_model.generate(audio=audio_path, user_prompt=full_prompt)
        item["ans"] = response
        tqdm.write(f"Processed {audio_path}, Answer: {response}")

    # Save the answers back to the JSON file
    with open(args.output_file, "w") as f:
        json.dump(example_data, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["ASR", "SER", "GR"],
        required=True,
        help="Task type: ASR (Automatic Speech Recognition), SER (Speech Emotion Recognition), or GR (Gender Recognition).",
    )
    parser.add_argument(
        "--constraint_type",
        "-c",
        type=str,
        choices=["chain-of-thought", "creative_writing"],
        required=True,
        help="Type of constraint to apply: chain-of-thought or creative_writing.",
    )
    parser.add_argument(
        "--subconstraint",
        "-s",
        type=str,
        choices=[
            "detectable_format:number_bullet_lists",
            "keywords:existence",
            "keywords:forbidden_words",
            "length_constraints:number_words",
            "length_constraints:number_sentences",
            "length_constraints:number_paragraphs",
        ],
        help="Sub-constraint type if constraint_type is creative_writing.",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=10,
        help="Number of examples to process. Default is 10.",
    )
    parser.add_argument(
        "--api_keys",
        type=str,
        default=None,
        help="Comma-separated Gemini API keys. If not provided, will use GEMINI_API_KEYS environment variable.",
    )
    parser.add_argument(
        "--metadata_file",
        "-m",
        type=str,
        default="./audios/CREMA-D_metadata.csv",
        help="Path to the metadata CSV file containing filename, transcription, emotion, and gender.",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="./generated_in_context_examples.json",
        help="Path to the input JSON file containing generated examples.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        help="Path to the output JSON file to save the answers. If not provided, will overwrite the input file.",
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file
        print("WARNING: Overwriting the input file with answers.")

    if args.subconstraint is None and args.constraint_type == "creative_writing":
        raise ValueError(
            "Sub-constraint must be provided when constraint_type is creative_writing."
        )

    if args.constraint_type == "chain-of-thought" and args.subconstraint is not None:
        raise ValueError(
            "Sub-constraint should not be provided when constraint_type is chain-of-thought."
        )

    main(args)

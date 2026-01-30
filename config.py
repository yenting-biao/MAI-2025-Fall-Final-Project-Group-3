import argparse

MAP_MODEL_NAME = {
    "qwen": "Qwen",
    "qwen2": "Qwen2",
    "desta2_5": "desta2_5",
    "blsp-emo": "BLSP-Emo",
    "qwen25_omni": "Qwen2.5-Omni",
}
MAP_AUDIO_TASK = {
    "ASR": "Automatic_speech_recognition",
    "SER": "Speech_emotion_recognition",
    "GR": "Gender_recognition",
    # "MMAU": "MMAU", # not implemented yet
}
IMPLEMENTED_IF_TASKS = [
    # closed_ended_questions
        "change_case:english_capital",
        "change_case:english_lowercase",
        "detectable_format:json_format",
        "startend:quotation",
        "detectable_format:title",
        "combination:repeat_prompt",
        "startend:end_checker",
   # creative_writing
        "detectable_format:number_bullet_lists",
        "keywords:existence",
        "keywords:forbidden_words",
        "length_constraints:number_words",
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
    # chain-of-thought
        "chain-of-thought",
]

TEST_SAMPLE = {
    "audio_path": "./data/audios/Automatic_speech_recognition/7176-92135-0019.flac",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
}

def get_task_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Model and Task Settings
    parser.add_argument("--model_name", type=str, default="qwen",
                        choices=["qwen", "qwen2", "desta2_5", "blsp-emo", "qwen25_omni"],
                        help="Name of the pre-trained language model to use.")

    parser.add_argument("--audio_task", type=str, default="ASR",
                        choices=["ASR", "SER", "GR", "MMAU"], # MMAU is not implemented yet
                        help="The specific audio-related task.")

    parser.add_argument(
        "--response_task", type=str, default="closed_ended_questions",
        choices=[
            "closed_ended_questions",
            "chain-of-thought",
            "creative_writing",
        ], help="The specific task for in-context learning.")

    parser.add_argument(
        "--IF_task", type=str, default=None,
        choices=[
            # closed_ended_questions
                "change_case:english_capital",
                "change_case:english_lowercase",
                "detectable_format:json_format",
                "startend:quotation",
                "detectable_format:title",
                "combination:repeat_prompt",
                "startend:end_checker",
            # creative_writing
                "detectable_format:number_bullet_lists",
                "keywords:existence",
                "keywords:forbidden_words",
                "length_constraints:number_words",
                "length_constraints:number_sentences",
                "length_constraints:number_paragraphs",
            # chain-of-thought
                "chain-of-thought",
        ], help="The format constraint task (i.e., instruction) for the model's response.")

    parser.add_argument("--examples", type=int, default=5, help="Number of in-context examples to use. Select from [0, 8]")

    return parser

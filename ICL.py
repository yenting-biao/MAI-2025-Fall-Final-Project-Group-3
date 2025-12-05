import argparse
import torch
import json, os
from typing import Dict
from models.basemodel import BaseModel

MAP_MODEL_NAME = {
    "qwen": "Qwen",
    "qwen2": "Qwen2",
    "desta2": "DeSTA2",
    "blsp-emo": "BLSP-Emo",
}
MAP_AUDIO_TASK = {
    "ASR": "Automatic_speech_recognition",
    "SER": "Speech_emotion_recognition",
    "GR": "Gender_recognition",
    # "MMAU": "MMAU", # not implemented yet
}
IMPLMENTED_IF_TASKS = [
    # closed_ended_questions
        "change_case:english_capital",
        "change_case:english_lowercase",
        "detectable_format:json_format",
        "startend:quotation",
        "detectable_format:title",
        "combination:repeat_prompt",
        "startend:end_checker",
#    # creative_writing (not implemented yet)
#         "detectable_format:number_bullet_lists",
#         "keywords:existence",
#         "keywords:forbidden_words",
#         "length_constraints:number_words",
#         "length_constraints:number_sentences",
#         "length_constraints:number_paragraphs",
]

# TEST_EXAMPLE = {
#     "audio_path": "./samples/sd-qa_1008642825401516622.wav",
#     # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
#     "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
#     "answer": "None"
# }

TEST_EXAMPLE = {
    "audio_path": "./in-context-examples/audios/general/11.wav",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}

# "common_voice_en_31703154.mp3", "instruction": "Tell the gender of the speaker from this audio recording. Choose the answer from \"Man\" or \"Woman\""

TEST_EXAMPLE = {
    "audio_path": "./data/audios/Automatic_speech_recognition/61-70968-0011.flac",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "Now answer the question: what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}


TEST_EXAMPLE = {
    "audio_path": "./data/audios/Gender_recognition/common_voice_en_31703154.mp3",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "Tell the gender of the speaker from this audio recording. Choose the answer from \"Man\" or \"Woman\"\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}

TEST_EXAMPLE = {
    "audio_path": "./data/audios/Gender_recognition/common_voice_en_17260337.mp3",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}

TEST_EXAMPLE = {
    "audio_path": "./data/audios/Automatic_speech_recognition/7176-92135-0019.flac",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}


def load_model(model_name, device: str = "cuda") -> BaseModel:
    if device == "cuda" and not torch.cuda.is_available():
        print("\033[41mCUDA is not available. Using CPU instead.\033[0m")
        device = "cpu"
    match model_name.lower():
        case "qwen":
            from models.Qwen import Qwen_Audio_Chat
            return Qwen_Audio_Chat(device=device)
        case "qwen2":
            from models.Qwen2 import Qwen2_Audio_Chat
            return Qwen2_Audio_Chat(device=device)
        case "desta2":
            from models.DeSTA2 import DeSTA2
            return DeSTA2(device=device)
        case "blsp-emo":
            from models.blsp_emo import BLSP_Emo
            return BLSP_Emo(device=device)
    raise ValueError(f"Model {model_name} not supported.")

def GenerateICLandTestExamples(icl_examples:list[dict], icl_audio_path:str, test_example: Dict[str, str], debug: bool = False) -> list[dict]:
    '''
        Generate In-Context Learning Examples and concatenate with test example.
        Output format:
            List[Dict] :
            [
                {
                    "audio_path": ...
                    "instruction": ...
                    "answer": ...
                }, ...
                {
                    "audio_path": ...
                    "instruction": ...
                    "answer": None              <-- For test IF effect
                }
            ]
    '''
    ret = []
    for item in icl_examples:
        ICL_example = {}
        ICL_example["audio_path"] = os.path.join(icl_audio_path, item["audio_path"])
        ICL_example["instruction"] = item["instruction"]
        ICL_example["answer"] = f" [ANS] {item.get('ans', None)}"
        ret.append(ICL_example)
        if debug:
            print(f"ICL Example added: {ICL_example}")
    # Insert test example at the end
    ret.append(test_example)
    return ret

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning (ICL) Configuration")

    # Model and Task Settings
    parser.add_argument("--model_name", type=str, default="qwen",
                        choices=["qwen", "qwen2", "desta2", "blsp-emo"],
                        help="Name of the pre-trained language model to use.")

    parser.add_argument("--audio_task", type=str, default="ASR",
                        choices=["ASR", "SER", "GR", "MMAU"], # MMAU is not implemented yet
                        help="The specific audio-related task.")

    parser.add_argument(
        "--response_task", type=str, default="closed_ended_questions",
        choices=[
            "closed_ended_questions",
            "chain-of-thought", # not implemented yet
            "creative_writing", # not implemented yet
        ], help="The specific task for in-context learning.")

    parser.add_argument(
        "--IF_task", type=str, default="change_case:english_capital",
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
        ], help="The format constraint task (i.e., instruction) for the model's response.")

    # ICL Settings
    parser.add_argument("--examples", type=int, default=5, help="Number of in-context examples to use. Select from [0, 8]")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    # Data Settings
    parser.add_argument("--icl_json_path", type=str, default="./in-context-examples/ICL_examples_processed.json", help="Path to the JSON file containing in-context examples.")
    parser.add_argument("--icl_audio_dir", type=str, default="./in-context-examples/audios/", help="Directory containing audio files for in-context examples.")
    parser.add_argument("--test_audio_dir", type=str, default="./data/audios/", help="Path to the audio files for the test cases.")

    args = parser.parse_args()
    return args

def verify_args(args: argparse.Namespace) -> None:
    if args.examples < 0 or args.examples > 8:
        raise ValueError("Number of in-context examples must be in [0, 8].")
    if args.model_name.lower() not in MAP_MODEL_NAME:
        raise ValueError(f"Model name {args.model_name} is not supported.")
    if args.audio_task.upper() not in MAP_AUDIO_TASK:
        raise ValueError(f"Audio task {args.audio_task} is not supported.")
    if args.response_task not in ["closed_ended_questions", "chain-of-thought", "creative_writing"]:
        raise ValueError(f"Response task {args.response_task} is not supported.")
    if args.IF_task not in IMPLMENTED_IF_TASKS:
        raise ValueError(f"IF task {args.IF_task} is not implemented yet.")
    if args.verbose:
        print("Arguments verified successfully.")

def main(args: argparse.Namespace) -> int:
    # Load Json file
    with open(args.icl_json_path, "r") as f:
        InContextDataset = json.load(f)
    icl_examples = InContextDataset[args.audio_task][args.response_task][args.IF_task][:args.examples]
    ICLexamples = GenerateICLandTestExamples(icl_examples, args.icl_audio_dir, TEST_EXAMPLE, debug=args.debug)

    # Load model
    model = load_model(args.model_name)
    print(f"\033[92m{MAP_MODEL_NAME[args.model_name.lower()]} model initialized.\033[0m")

    # Process input
    model.process_input(ICLexamples)
    if args.debug:
        print("-- Input processed. ---")
        print(f"\033[93m{model.messages}\033[0m")

    #   Generate response
    response = model.generate()
    print("Model response: ", f"\033[92m{response}\033[0m")
    return 0

if __name__ == "__main__":
    args = parse_args()
    verify_args(args)
    main(args)













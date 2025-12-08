import argparse
import torch
import json, os, datetime
import random
import numpy as np
from typing import Dict, Tuple

from pathlib import Path
from tqdm import tqdm

from models.basemodel import BaseModel

MAP_MODEL_NAME = {
    "qwen": "Qwen",
    "qwen2": "Qwen2",
    "desta2_5": "desta2_5",
    "blsp-emo": "BLSP-Emo",
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
#    # creative_writing (not implemented yet)
#         "detectable_format:number_bullet_lists",
#         "keywords:existence",
#         "keywords:forbidden_words",
#         "length_constraints:number_words",
#         "length_constraints:number_sentences",
#         "length_constraints:number_paragraphs",
]

TEST_SAMPLE = {
    "audio_path": "./data/audios/Automatic_speech_recognition/7176-92135-0019.flac",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
}

def set_seed(seed: int = 42, verbose: bool = False) -> None:
    # Python & OS
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch: CPU & GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if verbose:
        print(f"Seed set to {seed}")

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
        case "desta2_5":
            from models.DeSTA2_5 import DeSTA2_5
            return DeSTA2_5(device=device)
        case "blsp-emo":
            from models.blsp_emo import BLSP_Emo
            return BLSP_Emo(device=device)
    raise ValueError(f"Model {model_name} not supported.")

def GetICLData(args: argparse.Namespace) -> list[dict]:
    '''
        Load ICL examples from JSON file.
        Output format:
            List[Dict] :
            [
                {
                    "audio_path": ...
                    "instruction": ...
                    "ans": ...
                }, ...
            ]
    '''
    with open(args.icl_json_path, "r") as f:
        InContextDataset = json.load(f)
    return InContextDataset[args.audio_task][args.response_task][args.IF_task]

def GetTestCases(args: argparse.Namespace, audio_task_mapped: str) -> tuple[list[dict], str]:
    '''
        Load test cases from file or use predefined test sample.
        Output format:
            List[Dict] :
            [
                {
                    "id": ...
                    "audio_filepath": ...
                    "textual_audio": ...
                    "instruction": ...
                }, ...
            ]
    '''
    if args.use_test_sample:
        test_dir = ""
        test_audio_dir = ""
        test_cases = [TEST_SAMPLE]
        if args.debug:
            print(f"Using predefined test sample: {test_cases[0]}")
        return test_cases, test_audio_dir

    if args.test_eval_dir:
        test_audio_dir = args.test_audio_dir
        test_dir = args.test_eval_dir
        test_fn = os.path.join(test_dir, f"{args.response_task}.jsonl")
        with open(test_fn, "r") as fin:
            test_cases_tmp = [json.loads(line) for line in fin.readlines()]
        test_cases = []
        for tc in test_cases_tmp:
            condition_1 = tc["audio_filepath"].startswith(audio_task_mapped)
            condition_2 = tc["dataset"] == audio_task_mapped
            condition_3 = tc["instruction_id_list"][0] == args.IF_task
            if condition_1 and condition_2 and condition_3:
                test_cases.append(tc)

    # use test_audio_dir if test_eval_dir not specified
    else:
        test_audio_dir = os.path.join(args.test_audio_dir, audio_task_mapped)
        test_fn = os.path.join(test_audio_dir, "manifest.jsonl")
        with open(test_fn, "r") as fin:
            test_cases = [json.loads(line) for line in fin.readlines()]

    return test_cases, test_audio_dir

def GetOutputFilePath(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) / args.model_name.lower() / args.audio_task / args.response_task
    output_dir = output_dir / args.IF_task.replace(':', '_')
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_fn = output_dir / f"output_{args.examples}-shot_{ts}.jsonl"
    return output_fn

def GenerateICLandTestExamples(
    icl_data:list[dict],
    icl_audio_path:str,
    test_case_formatted: Dict[str, str],
    debug: bool = False
) -> list[dict]:
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
                    # no "answer" key for the test (query) example
                }
            ]
    '''
    ret = []
    for item in icl_data:
        ICL_example = {}
        ICL_example["audio_path"] = os.path.join(icl_audio_path, item["audio_path"])
        ICL_example["instruction"] = item["instruction"]
        ICL_example["answer"] = f" [ANS] {item.get('ans', None)} "
        ret.append(ICL_example)
        if debug:
            print(f"ICL Example added: {ICL_example}")

    # Insert test example at the end
    ret.append(test_case_formatted)
    return ret

def GenerateMessagesResponse(
    test_audio_dir: str,
    test_case: dict,
    model: BaseModel,
    icl_data: list[dict],
    icl_audio_dir: str,
    use_test_sample: bool = False,
    debug: bool = False,
) -> Tuple[str, str]:
    test_case_formatted = {
        "audio_path": os.path.join(test_audio_dir, test_case["audio_filepath"]),
        "instruction": test_case["instruction"],
    } if not use_test_sample else test_case
    conversation = GenerateICLandTestExamples(icl_data, icl_audio_dir, test_case_formatted, debug)
    model.process_input(conversation)
    if debug:
        print("-- Input processed. ---")
        print(f"\033[93m{model.messages}\033[0m")

    # Generate response
    response = model.generate()
    messages_str = json.dumps(model.messages, ensure_ascii=False)

    return messages_str, response

def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning (ICL) Configuration")

    # Model and Task Settings
    parser.add_argument("--model_name", type=str, default="qwen",
                        choices=["qwen", "qwen2", "desta2_5", "blsp-emo"],
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    parser.add_argument("--debug_examples", type=int, default=10, help="Number of test examples to run in debug mode.")


    # Dir Settings
    parser.add_argument("--output_dir", type=str, default="./model_responses/", help="Directory to save the outputs.")
    parser.add_argument("--icl_json_path", type=str, default="./in-context-examples/ICL_examples_processed.json", help="Path to the JSON file containing in-context examples.")
    parser.add_argument("--icl_audio_dir", type=str, default="./in-context-examples/audios/", help="Directory containing audio files for in-context examples.")

    """
    [IMPORTANT] Test Settings
    - If you want to test with a single predefined test sample instead of loading from a file,
      use the `--use_test_sample` flag, which overrides the test_audio_dir and test_eval_dir settings.
    - If you don't specify `test_eval_dir`, `test_audio_dir` would be treated as the source for test samples.
    """
    parser.add_argument("--test_audio_dir", type=str, default="./data/audios/", help="Path to the audio files for the test cases.")
    parser.add_argument("--test_eval_dir", type=str, default="./data/eval_data/", help="Path to the eval jsonl files for the test cases.")
    parser.add_argument("--use_test_sample", action="store_true", help="Use a single predefined test sample instead of loading from file to test.")

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
    if args.IF_task not in IMPLEMENTED_IF_TASKS:
        raise ValueError(f"IF task {args.IF_task} is not implemented yet.")
    if args.debug and args.debug_examples <= 0:
        raise ValueError("Number of debug examples must be greater than 0 when debug mode is enabled.")
    if args.verbose:
        print("Arguments verified successfully.")

def main(args: argparse.Namespace) -> None:
    t0 = datetime.datetime.now()
    audio_task_mapped = MAP_AUDIO_TASK[args.audio_task.upper()]
    print(f"\n\033[92mStarting ICL inference with model: {MAP_MODEL_NAME[args.model_name.lower()]}\n"
          f"audio task: {audio_task_mapped}\n"
          f"response task: {args.response_task}\n"
          f"IF task: {args.IF_task}\n"
          f"using {args.examples} in-context examples.\n"
          f"Starts at {t0.strftime('%Y-%m-%d %H:%M:%S')}\033[0m\n")

    # Load model
    model = load_model(args.model_name)
    print(f"\033[92m{MAP_MODEL_NAME[args.model_name.lower()]} model initialized.\033[0m")

    # Prepare ICL data and test cases
    icl_data = GetICLData(args) if args.examples > 0 else []
    test_cases, test_audio_dir = GetTestCases(args, audio_task_mapped)
    if args.debug:
        test_cases = test_cases[:min(args.debug_examples, len(test_cases))]
    if args.verbose:
        print(f"\033[93mLoaded {len(icl_data)} ICL examples and {len(test_cases)} test case(s).\033[0m")

    # Process each test case, generate a response, and save the response with metadata
    output_fn = GetOutputFilePath(args)
    print("\n\nStarting inference on test cases...\n")
    pbar = enumerate(test_cases) if args.debug or args.verbose else enumerate(tqdm(test_cases))
    with open(output_fn, "w") as fout:
        for i, test_case in pbar:
            set_seed(args.seed + i, args.verbose)
            icl_data_shuffled = icl_data.copy()
            random.shuffle(icl_data_shuffled)
            icl_data_examples = icl_data_shuffled[:args.examples]
            messages, response = GenerateMessagesResponse(
                test_audio_dir, test_case, model, icl_data_examples,
                args.icl_audio_dir, args.use_test_sample, args.debug
            )
            if args.debug or args.verbose:
                print(f"Model response [{i}]: \033[92m{response}\033[0m")
            output_data = {**test_case, "messages": messages, "response": response,}
            fout.write(json.dumps(output_data) + "\n")

    t1 = datetime.datetime.now()
    print(f"\033[92mAll responses saved to {output_fn}.\n"
          f"Ends at {t1.strftime('%Y-%m-%d %H:%M:%S')}\n"
          f"Time cost: {t1 - t0}.\033[0m\n")

if __name__ == "__main__":
    args = parse_args()
    verify_args(args)
    main(args)

import argparse, json, os, datetime, random, copy
from typing import Dict, Tuple
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from models.basemodel import BaseModel
from transformers import set_seed as set_transformers_seed

from config import get_task_parser
from config import MAP_MODEL_NAME, MAP_AUDIO_TASK, IMPLEMENTED_IF_TASKS, TEST_SAMPLE

#   Load MMAU audio information
MMAU_AUDIO_INFO = json.load(open("./in-context-examples/mmau-id2task.json", "r"))
MMAU_MINI_AUDIO_INFO = json.load(open("./in-context-examples/mmau-mini-id2task.json", "r"))

# Load audio2text file
AUDIO2TEXT = json.load(open("./audio_caption/audio_captions.json", "r"))

def _get_kwarg(test_case: dict, key: str):
    """Extract a value from test_case['kwargs'] which is a list[dict]."""
    for d in test_case.get("kwargs", []) or []:
        if key in d:
            return d[key]
    return None

def _rewrite_repeat_prompt_ans(ans, prompt_to_repeat: str) -> str:
    s = ans if isinstance(ans, str) else str(ans)
    # Keep everything after the first ":" as the transcription/content.
    if ":" in s:
        content = s.split(":", 1)[1].lstrip()
    else:
        content = s.strip()

    # Join without breaking formatting
    if prompt_to_repeat.endswith((": "," ","\n","\t")):
        return f"{prompt_to_repeat}{content}"
    return f"{prompt_to_repeat} {content}"

def _rewrite_end_checker_ans(ans, end_phrase: str, mmau: bool = False) -> str:
    s = ans if isinstance(ans, str) else str(ans)
    if mmau:
        first_line = s.split("This")[0].rstrip() if "This" in s else s.rstrip()
        first_line = f"{first_line}." if not first_line.endswith(".") else first_line
        return f"{first_line} {end_phrase}"
    else:
        splitlines = s.splitlines()
        first_line = splitlines[0].rstrip() if splitlines else s.rstrip()
        return f"{first_line}\n{end_phrase}"

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
    torch.use_deterministic_algorithms(True, warn_only=True)

    set_transformers_seed(seed)

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
        case "qwen25_omni":
            from models.Qwen25_omni import Qwen25_omni
            return Qwen25_omni(device=device)
        case "cascade_llama-3_1-8b-instruct":
            from models.cascade import CascadeModel
            return CascadeModel(llm_model_name="meta-llama/Llama-3.1-8B-Instruct", device=device)
        case "cascade_qwen-7b-chat":
            from models.cascade import CascadeModel
            return CascadeModel(llm_model_name="Qwen/Qwen-7B-Chat", device=device)
        case "cascade_qwen25-7b-instruct":
            from models.cascade import CascadeModel
            return CascadeModel(llm_model_name="Qwen/Qwen2.5-7B-Instruct", device=device)
        case model_name if "gemini" in model_name.lower():
            from models.Gemini import Gemini
            if model_name.lower() == "gemini":
                # Use default Gemini model
                return Gemini()
            else:
                generation_config = None
                if "no-thinking" in model_name.lower():
                    # assume model_name format is "{gemini_model_name}_no-thinking"
                    model_name = model_name.lower().replace("_no-thinking", "")
                    from google.genai import types
                    generation_config = {
                        "temperature": 1.0,
                        "thinking_config": types.ThinkingConfig(thinking_budget=0),
                    }
                return Gemini(model_name=model_name.lower(), generation_config=generation_config)
    raise ValueError(f"Model {model_name} not supported.")

def GetICLData(args: argparse.Namespace, max_examples: int = 8) -> list[dict]:
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
    if (args.audio_task == "MMAU"):
        IclData = InContextDataset[args.audio_task][args.response_task]["speech"]
    elif args.response_task == "chain-of-thought":
        IclData = InContextDataset[args.audio_task][args.response_task]
    else: # closed_ended_questions or creative_writing
        IclData = InContextDataset[args.audio_task][args.response_task][args.IF_task]

    if (args.audio_task == "MMAU") :
        return IclData

    # Verify ICL data
    assert len(IclData) == max_examples, \
        f"ICL data does not have the required number of examples: expected {max_examples}, got {len(IclData)}."
    assert all(item.get("audio_path") and item.get("instruction") and item.get("ans") for item in IclData), \
        "ICL data is not properly formatted: missing or empty 'audio_path', 'instruction', or 'ans' fields."

    return IclData

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
        test_fn = os.path.join(test_dir, f"{args.response_task}_corrected_filtered.jsonl" if args.response_task != "creative_writing" else f"{args.response_task}.jsonl")
        with open(test_fn, "r") as fin:
            test_cases_tmp = [json.loads(line) for line in fin.readlines()]
        test_cases = []
        for tc in test_cases_tmp:
            condition_1 = tc["audio_filepath"].startswith(audio_task_mapped)
            condition_2 = tc["dataset"] == audio_task_mapped
            condition_3 = tc["instruction_id_list"][0] == args.IF_task if args.response_task != "chain-of-thought" else True
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
    model_name_part = args.model_name.lower()
    output_dir = Path(args.output_dir) / Path(model_name_part) / args.audio_task / args.response_task
    output_dir = output_dir / args.IF_task.replace(':', '_')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_fn = output_dir / f"output_{args.examples}-shot.jsonl"
    return output_fn

def remove_output_constraints_from_instruction(instruction: str) -> str:
    tail_to_remove = "For example:\n```JSON\n{\n...\n}\n```"
    instruction = instruction.strip()
    if instruction.endswith(tail_to_remove):
        instruction = instruction[: -len(tail_to_remove)]

    assert instruction.count("\n") == 1, "Instruction does not have exactly 1 newline as expected."

    # Split instruction into two parts
    instruction = instruction.split("\n")[0]  # Keep only the part before the newline
    return instruction.strip()

def GenerateICLandTestExamples(
    icl_data:list[dict],
    icl_audio_path:str,
    test_case_formatted: Dict[str, str],
    debug: bool = False,
    remove_output_constraints: bool = False,
    no_audio_icl: bool = False,
    model_name: str = "",
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

    def info2String(info: list[dict]) -> str:
        audio_str = ""
        for i in info:
            audio_str += f"{i['time_segment']} {i['ASR']} (Gender: {i['Gender']}, Emotion: {i['Emotion']}).\n"
        return audio_str


    def wrap_audio_info_with_special_tokens(instruction: str, audio_info: str, model_name: str) -> str:
        audio_special_token_map = {
            "qwen2": ("<|audio_bos|>", "<|audio_eos|>"),
            "qwen25_omni": ("<|audio_bos|>", "<|audio_eos|>"),
            "desta2_5": ("<start_audio>", "<end_audio>"),
            "gemini-2.5-flash": ("<start_of_audio>", "<end_of_audio>"),
            "gemini-2.5-flash_no-thinking": ("<start_of_audio>", "<end_of_audio>"),
        }
        if model_name in audio_special_token_map:
            bos_token, eos_token = audio_special_token_map[model_name]
            return f"{bos_token}\n{audio_info}\n{eos_token}\n{instruction}"
        elif model_name == "blsp-emo":
            return f"{instruction}\n\nSpeech: {audio_info}"
        else:
            raise ValueError(f"Model {model_name} does not have defined audio special tokens.")

    ret = []
    for item in icl_data:
        ICL_example = {}

        ICL_example["audio_path"] = os.path.join(icl_audio_path, item["audio_path"]) if not no_audio_icl else None
        instruction = item["instruction"] if not remove_output_constraints else remove_output_constraints_from_instruction(item["instruction"])

        if no_audio_icl:
            audio_id = item["audio_path"].split("/")[-1]
            audio_info = AUDIO2TEXT.get(audio_id, "")
            audio_str = info2String(audio_info) if audio_info else ""
            instruction = wrap_audio_info_with_special_tokens(instruction, audio_str, model_name)

        ICL_example["instruction"] = instruction

        ans = item["ans"]
        assert ans is not None, "Answer in ICL example cannot be None."
        ICL_example["answer"] = json.dumps(ans, ensure_ascii=False) if isinstance(ans, dict) else str(ans)

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
    remove_output_constraints: bool = False,
    no_audio_icl: bool = False,
    model_name: str = "",
) -> Tuple[str, str]:
    test_case_formatted = {
        "audio_path": os.path.join(test_audio_dir, test_case["audio_filepath"]),
        "instruction": test_case["instruction"],
    } if not use_test_sample else test_case
    conversation = GenerateICLandTestExamples(icl_data, icl_audio_dir, test_case_formatted, debug, remove_output_constraints, no_audio_icl, model_name)
    model.process_input(conversation)
    if debug:
        print("-- Input processed. ---")
        print(f"\033[93m{model.messages}\033[0m")

    # Generate response
    response = model.generate()
    messages_str = json.dumps(model.messages, ensure_ascii=False)

    return messages_str, response

def parse_args():
    # Model and Task Settings
    parser = get_task_parser()

    # ICL Settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging.")
    parser.add_argument("--debug_examples", type=int, default=10, help="Number of test examples to run in debug mode.")

    # Dir Settings
    parser.add_argument("--output_dir", type=str, default="./model_responses/", help="Directory to save the outputs.")
    parser.add_argument("--icl_json_path", type=str, default="./in-context-examples/ICL_examples.json", help="Path to the JSON file containing in-context examples.")
    parser.add_argument("--icl_audio_dir", type=str, default="./in-context-examples/audios/", help="Directory containing audio files for in-context examples.")

    # experiment to remove output constraints
    parser.add_argument("--no_output_constraints", action="store_true", help="Whether to remove output constraints in instructions for ICL experiment. Using this flag will output to a separate folder with '_no_constraints' suffix for analysis.")

    # experiment to use no audio examples in ICL
    parser.add_argument("--no_audio_icl", action="store_true", help="Whether to remove audio information in the ICL examples, leaving only the instructions and answers. This is to test how audio affects the ICL performance. Using this flag will output to a separate folder with '_no_audio_icl' suffix for analysis.")

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

#   MMAU function
def MMAU_Get_ICL_Tasks(audio_id: str) -> Tuple[str, str]:
    #   Check audio id
    #   audio_id format: "MMAU/{NAME}.wav" try to get {NAME}
    audio_id = audio_id.split('/')[-1]  # Get the file name
    audio_id = audio_id[:-4]  # Remove the .wav extension

    if audio_id in MMAU_AUDIO_INFO:
        task_info = MMAU_AUDIO_INFO[audio_id]
    elif audio_id in MMAU_MINI_AUDIO_INFO:
        task_info = MMAU_MINI_AUDIO_INFO[audio_id]
    else:
        raise ValueError(f"Audio ID {audio_id} not found in MMAU audio info.")
    main_task = task_info["category"]
    sub_task = task_info["sub-category"]
    return main_task, sub_task

def rewrite_ans(args, test_case, icl_data_examples: list[dict]) -> list[dict]:
    '''
    For certain IF tasks,
    we need to modify the ICL examples to remove or change specific output constraints in the answers
    to ensure a fair evaluation of the model's capabilities without those constraints.
    This is only done when --no_output_constraints flag is set and there are ICL examples to modify.
    '''
    if args.IF_task in ("combination:repeat_prompt", "combination_repeat_prompt"):
        prompt_to_repeat = _get_kwarg(test_case, "prompt_to_repeat")
        if prompt_to_repeat is not None:
            for ex in icl_data_examples:
                ex["ans"] = _rewrite_repeat_prompt_ans(ex["ans"], prompt_to_repeat)

    elif args.IF_task in ("startend:end_checker", "startend_end_checker"):
        end_phrase = _get_kwarg(test_case, "end_phrase")
        if end_phrase is not None:
            for ex in icl_data_examples:
                ex["ans"] = _rewrite_end_checker_ans(ex["ans"], end_phrase, mmau=(args.audio_task == "MMAU"))

    return icl_data_examples

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
    pbar = enumerate(test_cases) if args.debug or args.verbose else enumerate(tqdm(test_cases, dynamic_ncols=True))
    with open(output_fn, "w") as fout:
        for i, test_case in pbar:
            set_seed(args.seed + i, args.verbose)
            if args.no_output_constraints:
                test_case["instruction"] = remove_output_constraints_from_instruction(test_case["instruction"])
            if "gemini" in args.model_name:
                model.generation_config["seed"] = args.seed + i
            if (args.audio_task == "MMAU"):
                main_task, sub_task = MMAU_Get_ICL_Tasks(test_case["audio_filepath"])
                icl_data_shuffled = icl_data[main_task][sub_task][args.IF_task].copy() if args.examples > 0 else []
            else:
                icl_data_shuffled = icl_data.copy()
            random.shuffle(icl_data_shuffled)
            icl_data_examples = copy.deepcopy(icl_data_shuffled[:args.examples])

            #   Rewrite ICL answers if needed
            # if args.no_output_constraints and args.examples > 0:
            icl_data_examples = rewrite_ans(args, test_case, icl_data_examples)

            messages, response = GenerateMessagesResponse(
                test_audio_dir, test_case, model, icl_data_examples,
                args.icl_audio_dir, args.use_test_sample, args.debug,
                args.no_output_constraints, args.no_audio_icl, args.model_name
            )
            if args.debug or args.verbose:
                print(f"Model response [{i}]: \033[92m{response}\033[0m")
            output_data = {**test_case, "messages": messages, "response": response,}
            if "gemini" in args.model_name:
                del output_data["response"]
                output_data["thinking_summary"] = model.thinking_summary
                if args.IF_task == "chain-of-thought" and model.thinking_summary:
                    output_data["response"] = f'<thinking_summary>\n{output_data["thinking_summary"]}\n</thinking_summary>\n{response}'  # Insert thinking summary into response
                else:
                    # Reinsert to ensure "response" key comes after "thinking_summary"
                    output_data["response"] = response
            fout.write(json.dumps(output_data) + "\n")

    t1 = datetime.datetime.now()
    print(f"\033[92mAll responses saved to {output_fn}.\n"
          f"Ends at {t1.strftime('%Y-%m-%d %H:%M:%S')}\n"
          f"Time cost: {t1 - t0}.\033[0m\n")

if __name__ == "__main__":
    args = parse_args()
    verify_args(args)
    main(args)

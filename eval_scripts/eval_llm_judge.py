#!/usr/bin/env python

import json
import os
import sys
from typing import Dict, Any, List

from tqdm import tqdm

# Add project root (parent of eval_scripts) to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import VLLMInference  # uses Qwen/Qwen3-8B by default
from config import get_task_parser
from config import MAP_AUDIO_TASK, IMPLEMENTED_IF_TASKS

# ---------- Prompt builder ----------

def build_if_judge_prompt(
    instruction: str,
    response: str,
) -> str:
    """
    Ask Qwen to give a single instruction-following score in [0, 1].
    """
    prompt = f"""
You are grading how well a model response follows a natural-language instruction.

IMPORTANT:
- Focus ONLY on whether the response respects the *format, style, and explicit constraints* in the instruction.
  Examples of explicit constraints:
  - required number of items or bullet points,
  - required answer options (e.g., "answer with yes or no"),
  - requested output format (e.g., JSON, a single word, a list),
  - requested language (e.g., English only).
- IGNORE whether the content itself is factually correct or relevant.
- Look only at instruction-following.

You must output a single score in [0, 1], called "if_score":

- 1.0  = perfectly follows all explicit constraints.
- 0.8  = almost perfect; only tiny cosmetic issues.
- 0.6  = mostly follows; some noticeable issues but the main constraints are respected.
- 0.3  = only partially follows; major constraints are violated.
- 0.0  = does not follow the instruction at all.

You may use any real value in [0, 1], but it should reflect this rubric.

Your output MUST be a single valid JSON object, with exactly these fields:
- "if_score": a number in [0, 1]
- "reason": a short explanation in one sentence

Do NOT include any text before or after the JSON.
Do NOT wrap the JSON in markdown.
Return ONLY the JSON.

=== Instruction ===
{instruction}

=== Model Response ===
{response}
"""
    return prompt.strip()

# ---------- Helper to parse JSON from Qwen output ----------

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from the model output.
    Handles the case where Qwen produces <think>...</think>{...}.
    """
    # If Qwen is in "thinking" mode, strip everything before the last </think>.
    lower = text.lower()
    think_tag = "</think>"
    pos = lower.rfind(think_tag)
    if pos != -1:
        candidate = text[pos + len(think_tag):]
    else:
        candidate = text

    # Find first '{'
    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No '{{' found in model output: {repr(text[:200])}...")

    # Simple brace matching to find the end of the JSON object
    depth = 0
    end = None
    for i, ch in enumerate(candidate[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(f"No matching '}}' found in model output: {repr(text[:200])}...")

    json_str = candidate[start:end].strip()
    return json.loads(json_str)


# ---------- Main evaluation logic ----------

def evaluate_if_level_with_qwen(
    judge: VLLMInference,
    input_path: str,
    output_path: str,
    max_samples: int | None = None,
    instruction_key: str = "instruction",
    response_key: str = "response",
) -> None:
    """
    For each line in input_path (.jsonl), ask Qwen3 to judge how well the
    model response follows the instruction, and write an augmented .jsonl.

    New fields added per record:
        - "if_level":  "strict" | "loose" | "not_follow"
        - "if_score":  1.0 | 0.5 | 0.0
        - "if_reason": short explanation string
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        batch_records: List[Dict[str, Any]] = []
        batch_prompts: List[str] = []

        # prepare all prompts first
        for line_idx, line in enumerate(tqdm(fin, desc="Evaluating IF level")):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Optionally limit number of samples
            if max_samples is not None and line_idx >= max_samples:
                break

            instruction = record.get(instruction_key, "")
            response = record.get(response_key, "")

            prompt = build_if_judge_prompt(instruction, response)

            batch_records.append(record)
            batch_prompts.append(prompt)

        # Feed into vllm
        outputs = judge.generate_response(batch_prompts)

        # Process outputs
        for record, raw_out in zip(batch_records, outputs):
            try:
                verdict = extract_json_from_text(raw_out)
                if_score = float(verdict.get("if_score", 0.0))
                if_score = max(0.0, min(1.0, if_score))  # clamp just in case
                reason = verdict.get("reason", "")
            except Exception as e:
                if_score = 0.0
                reason = f"Failed to parse judge JSON: {e}"

            # Always log the continuous score
            record["if_judge_raw_output"] = raw_out
            record["if_score"] = if_score
            record["if_reason"] = reason

            # Optional: derive a categorical label for later analysis
            if if_score >= 0.8:
                level = "strict"
            elif if_score >= 0.4:
                level = "loose"
            else:
                level = "not_follow"
            record["if_level"] = level

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- CLI ----------

def parse_args():
    parser = get_task_parser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to input .jsonl (e.g., <model_name>/<audio_task>/<response_task>/<if_task>/output_<k>-shot_<timestamp>.jsonl).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output .jsonl with Qwen IF judgments.",
    )
    parser.add_argument(
        "--judge_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Qwen3 model name for vLLM (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional: limit number of samples for quick testing.",
    )
    parser.add_argument(
        "--instruction_key",
        type=str,
        default="instruction",
        help="Key for instruction field in input JSONL (default: 'instruction').",
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="response",
        help="Key for response field in input JSONL (default: 'response').",
    )

    return parser.parse_args()

def get_task_names(args):
    audio_task = MAP_AUDIO_TASK[args.audio_task]

    if args.response_task == "creative_writing":
        response_task = args.response_task
        if args.IF_task is None:
            if_tasks = [
                "detectable_format_number_bullet_lists",
                "length_constraints_number_words",
                "length_constraints_number_sentences",
                "length_constraints_number_paragraphs",
            ]
            if args.audio_task == "ASR":
                if_tasks.append("keywords_existence")
                if_tasks.append("keywords_forbidden_words")
        else:
            if_task = args.IF_task
            if if_task not in IMPLEMENTED_IF_TASKS:
                raise ValueError(f"IF task {if_task} not implemented.")
            if if_task.startswith("keywords"):
                if args.audio_task != "ASR":
                    raise ValueError(f"IF task {if_task} only supported for ASR audio task.")
            if_tasks = [if_task]
    elif args.response_task == "chain-of-thought":
        response_task = args.response_task
        if args.IF_task is None or args.IF_task == "chain-of-thought":
            if_tasks = ["chain-of-thought"]
        else:
            raise ValueError(f"IF task {args.IF_task} not supported for chain-of-thought.")
    elif args.response_task == "closed_ended_questions":
        raise ValueError("closed_ended_questions is not supported in eval_llm_judge.py")
    else:
        raise ValueError(f"Unknown response task: {args.response_task}")

    return audio_task, response_task, if_tasks


def main(args):
    # Initialize the Qwen3 judge
    judge = VLLMInference(model_name=args.judge_name)

    if args.input_path and args.output_path:
        evaluate_if_level_with_qwen(
            judge=judge,
            input_path=args.input_path,
            output_path=args.output_path,
            max_samples=args.max_samples,
            instruction_key=args.instruction_key,
            response_key=args.response_key,
        )
        exit(0)

    test_model = args.model_name
    audio_task, response_task, if_tasks = get_task_names(args)
    print(f"Evaluating IF level for model={test_model}, audio_task={audio_task}, "
          f"response_task={response_task},\nif_tasks={if_tasks}")

    for if_task in if_tasks:
        print(f"\nEvaluating IF task: {if_task}")
        if_task_formatted = if_task.replace(":", "_")
        input_dir = f"model_responses/{test_model}/{audio_task}/{response_task}/{if_task_formatted}"
        output_dir = f"model_responses/{test_model}/{audio_task}/{response_task}/{if_task_formatted}/reports"
        os.makedirs(output_dir, exist_ok=True)

        for input_file in os.listdir(input_dir): # iterate over all output_{k}.jsonl, where k = 0, ..., 8
            if not input_file.startswith("output_") or not input_file.endswith(".jsonl"):
                continue
            input_path = os.path.join(input_dir, input_file)
            output_path = os.path.join(output_dir, f"judge@{input_file}")
            print(f"Input: {input_path}")
            print(f"Output: {output_path}")

            evaluate_if_level_with_qwen(
                judge=judge,
                input_path=input_path,
                output_path=output_path,
                max_samples=args.max_samples,
                instruction_key=args.instruction_key,
                response_key=args.response_key,
            )

if __name__ == "__main__":
    args = parse_args()
    main(args)

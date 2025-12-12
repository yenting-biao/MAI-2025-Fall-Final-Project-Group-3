# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import argparse
import collections
import dataclasses
import json
import os
import re
from typing import Dict, Optional, Sequence, Union, List
from pathlib import Path
from absl import flags
from absl import logging
from eval_scripts import instructions_registry
from config import get_task_parser

_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=False
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=False,
)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def _normalize_for_wer(text: str) -> List[str]:
    """Normalize text into a list of word tokens for WER.

    - Uppercase
    - Remove non-letter characters (except apostrophe)
    - Split on whitespace
    """
    text = text.upper()
    text = re.sub(r"[^A-Z']+", " ", text)
    tokens = text.split()
    return tokens


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Standard Levenshtein distance on token sequences."""
    n = len(ref)
    m = len(hyp)
    # dp[i][j]: distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def compute_wer(ref_text: str, hyp_text: str) -> float:
    """Compute word error rate (WER) between reference and hypothesis."""
    ref_tokens = _normalize_for_wer(ref_text)
    hyp_tokens = _normalize_for_wer(hyp_text)
    if not ref_tokens:
        return 0.0
    distance = _edit_distance(ref_tokens, hyp_tokens)
    return distance / float(len(ref_tokens))


def annotate_answer_correctness(result: dict, audio_task: str = "") -> dict:
    """Annotate each result dict with answer correctness.

    Adds:
      - 'wer' (float) when metric == 'wer'
      - 'exact_match' (bool)
      - 'answer_correct' (bool) as a generic flag
    """
    metric = result.get("metric")
    label = result.get("label")
    response = result.get("response", "")

    # Only handle examples with a label.
    if not metric or label is None:
        return result

    if metric == "wer":
        wer_value = compute_wer(label, response)
        result["wer"] = wer_value
        # "Correct" == perfect transcription by default
        exact = (wer_value == 0.0)
        result["answer_correct"] = exact

    elif metric == "accuracy":
        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().lower()

        correct = _norm(response) == _norm(label)
        result["answer_correct"] = correct

    else:
        # For 'open', 'keyword_exist', etc., we do not define correctness here.
        pass

    return result


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def read_key_to_prompt_dict(input_jsonl_filename):
  """Creates dictionary matching key to prompt."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      return_dict[example["key"]] = InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"])
  return return_dict


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(json.dumps(o, ensure_ascii=False))
      f.write("\n")


def test_instruction_following_strict(inp: dict, result: dict) -> dict[str, Union[bool, list[bool], str, list[str]]]:
    """Tests response to see if instructions are followed (strict)."""
    response = result["response"]
    instruction_list = inp["instruction_id_list"]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**inp["kwargs"][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp["prompt"])

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    # Original fields (for backward compatibility)
    result["follow_instruction_list"] = is_following_list
    result["follow_all_instructions"] = all(is_following_list)

    # Explicit "strict" aliases
    result["strict_follow_instruction_list"] = is_following_list
    result["strict_follow_all_instructions"] = all(is_following_list)

    return result


def test_instruction_following_loose_inplace(inp, result):
    """Upper bound for following instructions (looser matching).

    Tries multiple lightly-edited versions of the response:
    - removing first / last line
    - stripping '*' characters
    and marks an instruction as followed if ANY of these variants passes.
    """
    response = result["response"]

    # Generate relaxed variants of the response
    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()

    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")

    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    instruction_list = inp["instruction_id_list"]
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**inp["kwargs"][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp["prompt"])

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break
        is_following_list.append(is_following)

    result["loose_follow_instruction_list"] = is_following_list
    result["loose_follow_all_instructions"] = all(is_following_list)
    return result


def read_result_list(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  results = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      results.append(example)
  return results


def print_report(outputs):
  """Prints a report on accuracy scores."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  group_map = {
    "detectable_format:number_bullet_lists": "bullet_lists",
    "length_constraints:number_words": "length_constraints",
    "length_constraints:number_sentences": "length_constraints",
    "length_constraints:number_paragraphs": "length_constraints",
    "keywords:forbidden_words": "keywords",
    "keywords:existence": "keywords",
    "change_case:english_capital": "change_case",
    "change_case:english_lowercase": "change_case",
    "detectable_format:json_format": "json_format",
    "startend:quotation": "wrapping",
    "detectable_format:title": "wrapping",
    "combination:repeat_prompt": "startend",
    "startend:end_checker": "startend",
  }
  group_total = collections.defaultdict(int)
  group_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example["follow_instruction_list"]
    instruction_id_list = example["instruction_id_list"]

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      group = group_map.get(instruction_id, "other")
      group_total[group] += 1
      if followed_or_not:
        group_correct[group] += 1

  print(f"prompt-level: {prompt_correct / prompt_total}")
  print(f"instruction-level: {instruction_correct / instruction_total}")
  print()
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")

  print()
  print("===== Speech-IFEval accuracy =====")
  for group in sorted(group_total.keys()):
    accuracy = group_correct[group] / group_total[group]
    print(f"{group} {accuracy}")
  print(f"\nALL: {instruction_correct / instruction_total}")
  print("===================================")


def parse_args():
  """Parses command line arguments."""
  parser = get_task_parser()
  parser.add_argument(
      "--input_response_data",
      "-i",
      type=str,
      default=None,
      help="Path to input response data in JSONL format.",
  )
  return parser.parse_args()


def main():
    args = parse_args()

    if args.input_response_data:
      input_response_data = args.input_response_data
      input_file_name = input_response_data.split("/")[-1]
    else:
      input_response_data_dir = os.path.join("model_responses", args.model_name, args.audio_task, args.response_task)
      if args.response_task != "chain-of-thought" and args.IF_task:
        input_response_data_dir = os.path.join(input_response_data_dir, args.IF_task.replace(":", "_"))
      input_file_name = ""
      for file in os.listdir(input_response_data_dir):
        if file.startswith(f"output_{args.examples}-shot") and file.endswith(".jsonl"):
          if file > input_file_name:
            input_file_name = file
      if not input_file_name:
        raise ValueError("No response file found.")
      input_response_data = os.path.join(input_response_data_dir, input_file_name)

    results = read_result_list(input_response_data)
    # print(len(results))

    outputs = []
    output_file_name = f"rule_eval@{input_file_name}"
    logging.info("Generating %s...", output_file_name)

    for result in results:
        audio_task = result.get("dataset", "")
        condition = {
            "key": result["id"],
            "instruction_id_list": result["instruction_id_list"],
            "kwargs": result["kwargs"],
            # Use the per-example instruction as prompt for IFEval
            "prompt": result.get("instruction", ""),
        }

        # 1) Strict instruction following
        result = test_instruction_following_strict(condition, result)

        # 2) Loose instruction following
        result = test_instruction_following_loose_inplace(condition, result)

        # 3) Answer correctness (WER / exact match)
        result = annotate_answer_correctness(result, audio_task)

        outputs.append(result)

    # Aggregate strict IF stats (as before).
    follow_all_instructions = [o["follow_all_instructions"] for o in outputs]
    IF_rate = sum(follow_all_instructions) / len(outputs)
    logging.info("Strict IF Rate: %f", IF_rate)

    (Path(input_response_data).parent / "reports").mkdir(
        parents=True, exist_ok=True
    )
    output_file_name = str(
        (Path(input_response_data).parent / "reports")
        / f"{output_file_name}"
    )
    write_outputs(output_file_name, outputs)
    logging.info("Generated: %s", output_file_name)

    # Prints instruction-following accuracy report (strict).
    # print("=" * 64)
    print(f"output_file_name: {output_file_name}")
    # print_report(outputs)
    # print(output_file_name)

    # Print simple semantic-accuracy summary for debugging.
    # wer_values = [o["wer"] for o in outputs if "wer" in o]
    # if wer_values:
    #     mean_wer = sum(wer_values) / len(wer_values)
    #     print(f"Mean WER over {len(wer_values)} examples: {mean_wer:.4f}")

    # answer_flags = [o["answer_correct"] for o in outputs if "answer_correct" in o]
    # if answer_flags:
    #     ans_acc = sum(answer_flags) / len(answer_flags)
    #     print(f"Answer-correct accuracy over labeled examples: {ans_acc:.4f}")


if __name__ == "__main__":
  main()

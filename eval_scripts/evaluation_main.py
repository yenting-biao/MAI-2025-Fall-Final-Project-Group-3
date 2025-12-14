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

import collections
import dataclasses
import json
import os
import re
from typing import Dict, Optional, Union, List
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


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Handles ```json ... ``` or ``` ... ```
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _strip_outer_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s


def _strip_answer_prefixes(s: str) -> str:
    """Strip common leading answer markers like '[ANS]' used in ICL examples."""
    s = (s or "").strip()
    # Examples: "[ANS] happy", "[Answer] Woman"
    s = re.sub(r"^\s*\[(?:ANS|ANSWER)\]\s*", "", s, flags=re.IGNORECASE)
    # Examples: "ANS: happy", "Answer - Woman"
    s = re.sub(r"^\s*(?:ANS|ANSWER)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _get_kwargs_for_instruction(result: dict, instruction_id: str) -> dict:
    """Return kwargs dict associated with a specific instruction_id, if present.

    Tolerates ':' vs '_' variants in instruction ids.
    """
    ids = result.get("instruction_id_list") or []
    kwargs_list = result.get("kwargs") or []
    target = instruction_id
    target_alt = instruction_id.replace(":", "_") if ":" in instruction_id else instruction_id.replace("_", ":")
    for iid, kw in zip(ids, kwargs_list):
        if iid == target or iid == target_alt or iid.replace(":", "_") == target.replace(":", "_"):
            return kw or {}
    return {}


def _has_instruction(result: dict, instruction_id: str) -> bool:
    """Check for instruction_id; tolerant to ':' vs '_' variants."""
    ids = set(result.get("instruction_id_list") or [])
    if instruction_id in ids:
        return True
    # tolerate underscore/colon variants
    ids_norm = set(i.replace(":", "_") for i in ids)
    return instruction_id.replace(":", "_") in ids_norm


def _try_parse_json_answer(s: str, audio_task: str) -> Optional[str]:
    """Try to parse a JSON object from s and extract the answer string."""
    s2 = _strip_outer_quotes(_strip_code_fences(s))

    obj = None
    try:
        obj = json.loads(s2)
    except Exception:
        # Try extracting the first {...} block (handles extra text)
        m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None

    key_priority = {
        "GR":  ["gender", "label", "answer", "prediction", "pred"],
        "SER": ["emotion", "label", "answer", "prediction", "pred"],
        "ASR": ["transcript", "text", "answer", "prediction", "pred"],
    }
    for k in key_priority.get(audio_task, []):
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fallback: first non-empty string value
    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def _remove_title_tokens(s: str) -> str:
    """Remove <<title>> tokens (used by detectable_format:title)."""
    # Remove all occurrences of <<...>>
    s2 = re.sub(r"<<[^\n]+>>", "", s)
    # Drop now-empty lines
    lines = [ln for ln in s2.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _strip_leading_list_prefix(s: str) -> str:
    """Strip common bullet/number list prefixes from the start of s."""
    return re.sub(r"^\s*(?:\d+\s*[\.|\)|:|-]\s*|[-*•]\s*)", "", s).strip()


def extract_answer_for_scoring(result: dict, audio_task: str = "") -> str:
    """Extract the semantic answer from a (possibly wrapped) response for correctness scoring."""
    response = (result.get("response") or "").strip()

    # Always strip code fences first (common with json_format).
    response = _strip_code_fences(response)

    # 1) JSON format: parse JSON and pull the answer field/value
    if _has_instruction(result, "detectable_format:json_format"):
        parsed = _try_parse_json_answer(response, audio_task)
        if parsed is not None:
            return parsed

    # 2) Quotation wrapper: strip outer quotes
    if _has_instruction(result, "startend:quotation"):
        response = _strip_outer_quotes(response)

    # 3) Title wrapper: remove <<...>> token(s)
    if _has_instruction(result, "detectable_format:title"):
        response = _remove_title_tokens(response)

    # 4) Repeat-prompt: remove repeated prompt prefix if available
    if _has_instruction(result, "combination:repeat_prompt"):
        kw = _get_kwargs_for_instruction(result, "combination:repeat_prompt")
        prompt_to_repeat = kw.get("prompt_to_repeat")
        if isinstance(prompt_to_repeat, str) and prompt_to_repeat.strip():
            p = prompt_to_repeat.strip()
            # Remove once at start (case-insensitive), including potential leading whitespace
            pattern = r"^\s*" + re.escape(p) + r"\s*"
            new_resp = re.sub(pattern, "", response, count=1, flags=re.IGNORECASE | re.DOTALL)
            if new_resp != response:
                response = new_resp.strip()
            else:
                # Fallback: drop first non-empty line if it matches
                lines = [ln for ln in response.splitlines() if ln.strip()]
                if lines and lines[0].strip().lower() == p.lower():
                    response = "\n".join(lines[1:]).strip()

    # 5) End-checker: remove forced ending phrase if available
    if _has_instruction(result, "startend:end_checker"):
        kw = _get_kwargs_for_instruction(result, "startend:end_checker")
        end_phrase = kw.get("end_phrase")
        if isinstance(end_phrase, str) and end_phrase.strip():
            ep = end_phrase.strip()
            # Compare case-insensitively
            resp_r = response.rstrip()
            if resp_r.lower().endswith(ep.lower()):
                response = resp_r[: len(resp_r) - len(ep)].rstrip()

    # 6) For numbered bullet list outputs, strip leading list marker on first line
    if _has_instruction(result, "detectable_format:number_bullet_lists"):
        lines = [ln for ln in response.splitlines() if ln.strip()]
        if lines:
            # Often the answer is the first bullet item
            response = _strip_leading_list_prefix(lines[0])
    response = _strip_answer_prefixes(response)
    return response.strip()



def annotate_answer_correctness(result: dict, audio_task: str = "") -> dict:
    """Annotate each result dict with answer correctness.

    IMPORTANT: correctness is computed on an extracted answer string that
    removes / parses IF wrappers (e.g., JSON, quotation marks, title tokens,
    repeated prompt, end phrase).

    Adds:
      - 'response_for_scoring' (str): extracted answer used for scoring
      - 'wer' (float) when metric == 'wer'
      - 'exact_match' (bool) when metric == 'wer' or 'accuracy'
      - 'answer_correct' (bool) as a generic flag
    """
    metric = result.get("metric")
    label = result.get("label")

    # Only handle examples with a label.
    if not metric or label is None:
        return result

    response_for_scoring = extract_answer_for_scoring(result, audio_task)
    result["response_for_scoring"] = response_for_scoring

    if metric == "wer":
        wer_value = compute_wer(label, response_for_scoring)
        result["wer"] = wer_value
        exact = (wer_value == 0.0)
        result["exact_match"] = exact
        # "Correct" == perfect transcription by default
        result["answer_correct"] = exact

    elif metric == "accuracy":
        def _norm(s: str) -> str:
            s = _strip_outer_quotes(_strip_code_fences(s))
            s = _strip_answer_prefixes(s)
            # Remove punctuation, keep letters/digits/underscore/space
            s = re.sub(r"[^\w\s]+", "", s)
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        norm_label = _norm(str(label))
        norm_resp = _norm(response_for_scoring)

        # If this looks like a small-label classification task (GR/SER), extract the predicted label token
        # from the response instead of requiring exact string equality.
        allowed_ser = {"neutral", "happy", "sad", "angry"}
        allowed_gr = {"man", "woman"}

        # Prefer using audio_task if provided; otherwise infer from label space.
        task = (audio_task or "").upper().strip()
        if (task == "SER") or (norm_label in allowed_ser):
            allowed = allowed_ser
        elif (task == "GR") or (norm_label in allowed_gr):
            allowed = allowed_gr
        else:
            allowed = set()

        if allowed:
            tokens = norm_resp.split()
            candidates = [t for t in tokens if t in allowed]
            pred = candidates[-1] if candidates else norm_resp
            correct = (pred == norm_label)
        else:
            correct = (norm_resp == norm_label)

        result["exact_match"] = correct
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

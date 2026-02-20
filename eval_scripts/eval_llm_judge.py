#!/usr/bin/env python

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import os
from dotenv import load_dotenv

from jiwer import wer
from tqdm import tqdm
from whisper_normalizer.basic import BasicTextNormalizer

from .utils import VLLMInference, OpenAIInference
from config import get_task_parser
from config import IMPLEMENTED_IF_TASKS


def normalize_text(text: str) -> str:
    normalizer = BasicTextNormalizer()
    normalized_text = text.replace("<", "").replace(">", "")
    return normalizer(normalized_text).strip()


def extract_result(text: str) -> str | None:
    pattern = r"(?i)(?<=result:\s)(yes|no)"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


def extract_all_text_after_result(text: str) -> str | None:
    pattern = r"(?i)(?<=result:\s)(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None


def textual_audio_to_label(textual_audio: str) -> dict:
    """
    return a dict: {
        "ASR": transcript,
        "SER": emotion,
        "GR": gender
    }
    example textual_audio:
    [00:00:00 - 00:00:05] Mary Taylor, however, related the tale of Zora to Mrs. Gray's private ear later.(Gender: Female, Emotion: neutral)
    """

    transcript = textual_audio.split("]")[-1].split("(")[0].strip()
    gender = textual_audio.split("Gender:")[-1].split(",")[0].strip()
    emotion = textual_audio.split("Emotion:")[-1].split(")")[0].strip()
    return {"ASR": transcript, "SER": emotion, "GR": gender}


def build_eval_prompt_components(
    data: Dict[str, Any], remove_instruction: bool = False
) -> tuple[str, str, str]:
    instruction = data.get("instruction", "")
    if remove_instruction:
        instruction = instruction.split("\n")[0]
    label = data.get("label", "")
    response = data.get("response", "")
    metric = data.get("metric")

    if metric == "accuracy":
        system_prompt = f"""You will be given a **question**, a corresponding **ground truth answer** and a **response** from a model. Model's response is a reply to the question. Your task is to judge if "model's response" aligns with the "ground Truth answer" based on the "question".

Evaluation criteria:
* Judge alignment based on semantic correctness, not surface-level wording.
* Minor paraphrasing or differences in expression are acceptable if the meaning is equivalent.
* If the model's response misses essential information, or contradicts the ground truth answer, it should be considered non-aligned.

Please strictly follow the guidelines below:
* First, provide a brief explanation why the response aligns or does not align with the ground truth answer, based on the criteria above.
* Then Output "YES" if the response aligns with the ground truth answer; output "NO" if the response does not match the ground truth answer.
* Answer in the following format exactly:

```
Explanation: <your explanation>
Result: <YES or NO>
```
"""
        content = (
            f"**Question**: {instruction}\n**Ground Truth Answer**: {label}\n**Model's Response**: "
            f"{response}"
        )
    elif metric == "wer":
        system_prompt = f"""You will be given a **question** and a **model's response**. The question asks the model to **transcribe audio into text (ASR)**. The model’s response may include explanations, reasoning, or meta-comments in addition to the transcription.

Your task is to extract the **ASR transcription only**.

**Output format requirements:**

You must output **exactly two lines** in the following format:

```
Explanation: <your explanation>
Result: <extracted ASR substring, do not wrap in quotes or delimiters>
```

**Extraction rules:**

* In `Explanation`, briefly describe how you identified the ASR transcription and removed non-ASR content.
* In `Result`, output the extracted ASR transcription only. No quotation marks or delimiters.
* The extracted text must be a **continuous substring copied verbatim** from the model’s response.
* Do **not** modify, normalize, reformat, or rewrite the text in any way.
* Remove all non-ASR content, including introductions, explanations, reasoning, or meta-language.
* Do **not** include quotation marks or any other delimiters around the ASR text.
* If the response does **not** contain any ASR transcription, leave `Result` **empty** (i.e., `Result:` followed by nothing).

The extracted substring in `Result` will be evaluated using the **WER metric**, so **exact character-level matching** is required. Do NOT wrap the extracted text in quotes or any delimiters.
"""
        content = f"**Question**: {instruction}\n**Model's Response**: {response}"
    elif metric == "cot":
        system_prompt = f"""You will be given a **user input** and a **model's response**. The model's response is a reply to the user input. Your task is to determine whether the response demonstrates **reasoning behavior anywhere in the response**, regardless of order or position.

**Reasoning behavior includes (but is not limited to):**

* Explicit analysis or commentary at the beginning (e.g., “Let’s analyze…”, “First, consider…”).
* Breaking the problem into parts or cases.
* Explaining intermediate steps, assumptions, or decision criteria.
* Justifying an answer, even if the final answer appears later or earlier.
* Meta-reasoning about how the answer is derived.
* Analysis of the question or problem before providing an answer.

For example, reasoning behavior may involve phrases like:
- "1. The audio contains... 2. ..."
- "To identify ..., we must first ..."
- "Let's break this down into steps..."
- "**Tone**: The tone of the speech is... **Pitch**: ..."


A response should be classified as **NO** only if it consists solely of a direct, minimal answer or factual statement, **without any explanation, justification, or analytical content anywhere in the response**. For example,

- "The original content of this audio is: '...'" (no reasoning, just a direct transcription)
- "The person speaking exhibits a happy mood." (no reasoning, just a direct statement)
- "The pair of words that meet this criteria is 'Erwin, director.'" (no reasoning, just a direct answer)


Please strictly follow the guidelines below:

* First, briefly explain why the response should be classified as demonstrating reasoning behavior or not, based on the criteria above.
* Then output "YES" if the response contains any reasoning behavior anywhere in the response.
* Output "NO" only if the response is entirely non-analytical.
* Answer in the following format exactly:

```
Explanation: <your explanation>
Result: <YES or NO>
```
"""
        content = f"**User input**: {instruction}\n**Model's Response**: {response}"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    prompt = f"{system_prompt}\n\n{content}"
    return prompt, system_prompt, content


BatchEntry = tuple[Dict[str, Any], str, str, str]


def flush_eval_batch(
    judge: VLLMInference | OpenAIInference, batch: List[BatchEntry], fout
) -> None:
    if not batch:
        return

    if isinstance(judge, VLLMInference):
        prompts = [entry[1] for entry in batch]
    elif isinstance(judge, OpenAIInference):
        prompts = [
            (entry[2], entry[3]) for entry in batch
        ]  # (system_prompt, user_prompt)
    else:
        raise ValueError("Unsupported judge type.")
    responses = judge.generate_response(prompts)
    for (data, prompt, system_prompt, user_prompt), response in zip(batch, responses):
        data["eval_messages"] = prompt
        data["eval_response"] = response
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        logging.info(json.dumps(data, ensure_ascii=False))
    batch.clear()


def run_llm_evaluation(args: Any, judge: VLLMInference | OpenAIInference) -> None:
    print(f"\nEvaluating LLM judge on {args.input_response_data}\n")
    input_response_data_path = Path(args.input_response_data)
    if args.task_level:
        output_dir = input_response_data_path.parent / "reports-task-level"
    else:
        output_dir = input_response_data_path.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_dir / f"{input_response_data_path.stem}.log",
        level=logging.INFO,
        force=True,
    )

    tmp_output_file = tmp_dir / f"{input_response_data_path.stem}.jsonl"
    output_file = output_dir / f"llm_eval@{input_response_data_path.stem}.jsonl"
    remove_instruction = input_response_data_path.stem in ["close", "close.1"]

    if args.stage < 1:
        logging.info("=================== LLM evaluation ====================")
        logging.info(f"Processing {input_response_data_path}")
        logging.info(f"Output file: {tmp_output_file}")
        with input_response_data_path.open("r") as fin, tmp_output_file.open(
            "w"
        ) as fout:
            datas = [json.loads(line) for line in fin.readlines() if line.strip()]

            if args.task_level:
                # if "ASR" in path, metric set to "wer"; else set to "accuracy"
                for data in datas:
                    if "ASR" in str(input_response_data_path):
                        data["metric"] = "wer"
                    else:
                        data["metric"] = "accuracy"

                    if "label" not in data:
                        metadata = textual_audio_to_label(data["textual_audio"])
                        if "ASR" in str(input_response_data_path):
                            data["label"] = metadata["ASR"]
                        elif "SER" in str(input_response_data_path):
                            data["label"] = metadata["SER"]
                        elif "GR" in str(input_response_data_path):
                            data["label"] = metadata["GR"]
                        else:
                            raise ValueError("Unknown audio task in path.")

            batch: List[BatchEntry] = []
            for data in tqdm(datas, desc="Generating LLM judgments"):
                prompt, system_prompt, content = build_eval_prompt_components(
                    data, remove_instruction
                )
                batch.append((data, prompt, system_prompt, content))
            flush_eval_batch(judge, batch, fout)

    if args.stage < 2:
        logging.info("=================== Performance Evaluation ====================")
        dataset_group: dict[str, list[int]] = defaultdict(list)
        hyps: List[str] = []
        refs: List[str] = []
        with tmp_output_file.open("r") as fin, output_file.open("w") as fout:
            datas = [json.loads(line) for line in fin.readlines() if line.strip()]
            for data in tqdm(datas, desc="Aggregating metrics"):
                metric = data.get("metric")
                dataset = data.get("dataset", "unknown")
                if metric == "accuracy":
                    result = extract_result(data.get("eval_response", "")).split(
                        "</think>"
                    )[-1]
                    correct = bool(result and result.lower() == "yes")
                    dataset_group[dataset].append(1 if correct else 0)
                    data["correct"] = correct
                elif metric == "wer":
                    result = extract_all_text_after_result(
                        data.get("eval_response", "").split("</think>")[-1].strip()
                    )
                    hyp = normalize_text(result if result is not None else "")
                    if "label" not in data:
                        raise ValueError(
                            "WER metric requires 'label' field in the data."
                        )
                    ref = normalize_text(data.get("label", "").strip())
                    hyps.append(hyp)
                    refs.append(ref)
                    data["correct"] = wer(reference=[ref], hypothesis=[hyp])

                elif metric == "cot":
                    result = extract_result(data.get("eval_response", ""))
                    correct = bool(result and result.lower() == "yes")
                    dataset_group["cot"].append(1 if correct else 0)
                    data["correct"] = correct
                else:
                    logging.warning(f"Skipping unsupported metric {metric}")
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

        if refs:
            wer_score = wer(reference=refs, hypothesis=hyps)
            logging.info(f"WER: {wer_score}")
            print(f"WER: {wer_score}")
        for dataset, corrects in dataset_group.items():
            if not corrects:
                continue
            accuracy = sum(corrects) / len(corrects)
            logging.info(f"{dataset} ACC: {accuracy}")
            print(f"{dataset} ACC: {accuracy}")


def get_task_names(args):
    audio_task = args.audio_task

    if args.response_task == "creative_writing":
        raise ValueError(
            "creative_writing is not supported in eval_llm_judge.py. It should be able to be judged directly with rule-based methods."
        )
        # response_task = args.response_task
        # if args.IF_task is None:
        #     if_tasks = [
        #         "detectable_format_number_bullet_lists",
        #         "length_constraints_number_words",
        #         "length_constraints_number_sentences",
        #         "length_constraints_number_paragraphs",
        #     ]
        #     if args.audio_task == "ASR":
        #         if_tasks.append("keywords_existence")
        #         if_tasks.append("keywords_forbidden_words")
        # else:
        #     if_task = args.IF_task
        #     if if_task not in IMPLEMENTED_IF_TASKS:
        #         raise ValueError(f"IF task {if_task} not implemented.")
        #     if if_task.startswith("keywords"):
        #         if args.audio_task != "ASR":
        #             raise ValueError(
        #                 f"IF task {if_task} only supported for ASR audio task."
        #             )
        #     if_tasks = [if_task]
    elif args.response_task == "chain-of-thought":
        response_task = args.response_task
        if args.IF_task is None or args.IF_task == "chain-of-thought":
            if_tasks = ["chain-of-thought"]
        else:
            raise ValueError(
                f"IF task {args.IF_task} not supported for chain-of-thought."
            )
    elif args.response_task == "closed_ended_questions":
        raise ValueError("closed_ended_questions is not supported in eval_llm_judge.py")
    else:
        raise ValueError(f"Unknown response task: {args.response_task}")

    return audio_task, response_task, if_tasks


def parse_args() -> argparse.Namespace:
    parser = get_task_parser()
    parser.add_argument(
        "--input_response_data",
        "-i",
        type=str,
        help="Path to the JSONL of responses to be judged (must include metric labels).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[0, 1],
        default=0,
        help="Pipeline stage (0: generate & save judge outputs; 1: aggregate metrics).",
    )
    parser.add_argument(
        "--judge_name",
        type=str,
        default="gpt-5-mini-2025-08-07",  # "Qwen/Qwen3-8B",
        help="Model name passed to LLM judge (default: gpt-5-mini-2025-08-07).",
    )
    parser.add_argument(
        "--task_level",
        action="store_true",
        help="Whether to do task-level evaluation. When not set, do IF evaluation.",
    )
    parser.add_argument(
        "--use_vllm_judge",
        action="store_true",
        help="Whether to use vLLMInference as judge. If not set, use OpenAIInference.",
    )
    parser.add_argument(
        "--no_output_constraints",
        action="store_true",
        help="Whether to judge the model responses with output constraints removed from the instructions.",
    )
    parser.add_argument(
        "--no_audio_icl",
        action="store_true",
        help="Whether to judge the model responses with audios removed from the in-context learning examples.",
    )
    parser.add_argument(
        "--audio_only",
        action="store_true",
        help="Whether to judge the model responses with only audio inputs and no textual instructions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_vllm_judge:
        judge = VLLMInference(model_name=args.judge_name)
    else:
        load_dotenv()  # Load environment variables from .env file
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        judge = OpenAIInference(model_name=args.judge_name, api_key=openai_api_key)

    if args.input_response_data:
        run_llm_evaluation(args, judge)
        exit(0)

    test_model = args.model_name
    audio_task, response_task, if_tasks = get_task_names(args)
    print(
        f"Evaluating IF level for model={test_model}, audio_task={audio_task}, "
        f"response_task={response_task},\nif_tasks={if_tasks}"
    )

    for if_task in if_tasks:
        print(f"\nEvaluating IF task: {if_task}")
        if_task_formatted = if_task.replace(":", "_")
        if args.audio_only:
            input_dir = f"model_responses_audio_only/{test_model.lower()}/{audio_task}/{response_task}/{if_task_formatted}"
        elif args.no_audio_icl and args.no_output_constraints:
            input_dir = f"model_responses_no_constraints_no_audio_icl/{test_model.lower()}/{audio_task}/{response_task}/{if_task_formatted}"
        elif args.no_output_constraints:
            input_dir = f"model_responses_no_constraints/{test_model.lower()}/{audio_task}/{response_task}/{if_task_formatted}"
        else:
            input_dir = f"model_responses/{test_model.lower()}/{audio_task}/{response_task}/{if_task_formatted}"

        # Check that there are exactly 9 output files and the filenames are in the foramat "output_{k}*.jsonl", where k = 0, ..., 8
        candidate_files = sorted(
            list(
                f
                for f in os.listdir(input_dir)
                if f.startswith("output_") and f.endswith(".jsonl")
            )
        )
        if len(candidate_files) == 8:
            # they should be output_1 to output_8, missing output_0
            # if not, raise error
            expected_files = [f"output_{k}-shot.jsonl" for k in range(1, 9)]
            assert (
                candidate_files == expected_files
            ), f"Expected files {expected_files}, found {candidate_files}."
            print(
                f"Found 8 output files (output_1 to output_8) in {input_dir}, with output_0 missing as expected. Proceeding with evaluation."
            )
        else:
            assert (
                len(candidate_files) == 9
            ), f"Expected 9 output files in {input_dir}, found {len(candidate_files)}."
            for i, file_name in enumerate(candidate_files):
                expected_prefix = f"output_{i}"
                assert file_name.startswith(
                    expected_prefix
                ), f"Expected file starting with {expected_prefix}, found {file_name}."

        for input_file in candidate_files:
            args.input_response_data = os.path.join(input_dir, input_file)
            run_llm_evaluation(args, judge)


if __name__ == "__main__":
    main()

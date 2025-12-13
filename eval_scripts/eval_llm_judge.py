#!/usr/bin/env python

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import Any, Dict, List

from jiwer import wer
from tqdm import tqdm
from whisper_normalizer.basic import BasicTextNormalizer

from utils import VLLMInference


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
        system_prompt = (
            "You will be given a question, a corresponding correct answer and a "
            "response from a model. Model's Response is a reply to the Question."
            ' Your task is to judge if "Model\'s Response" aligns with the "Ground'
            ' Truth Answer" based on the "Question". Please strictly follow the'
            ' guidelines below:\n- Answer with the format "Result: <YES or NO>" at the'
            ' end.\n- Output "YES" if the response aligns with the ground truth'
            ' answer; output "NO" if the response does not match the ground truth'
            " answer."
        )
        content = (
            f"Question: {instruction}\nGround Truth Answer: {label}\nModel's Response: "
            f"{response}"
        )
    elif metric == "wer":
        system_prompt = (
            "You will be given a response from an ASR model. Your task is to extract a "
            "substring from the model's response that eliminates all extra phrases, "
            "explanations, or introductory text. The substring will be evaluate by the WER"
            " metric, so it should be exactly the same as the model's response, with no"
            " modifications.\n\nPlease strictly follow the guidelines below:\n- The substring"
            " should be exactly the same as the model's response, with no modifications.\n- "
            "Eliminate all extra phrases, explanations, or introductory text while keeping"
            " the substring itself 100% unchanged.\n- You must output the substring only."
        )
        content = f"Question: {instruction}\nModel's Response: {response}"
    elif metric == "cot":
        system_prompt = (
            "You will be given a user input and a model response. The model's response is"
            " a reply to the user input. Your task is to determine whether the response"
            " demonstrates reasoning behavior, such as breaking down the problem, explaining"
            " intermediate steps, or providing an analysis.\n\nPlease strictly follow"
            ' the guidelines below:\n- Output "YES" if the response includes any form of behavior beyond'
            ' a direct answer corresponding to the user input.\n- Output "NO" only if'
            " the response is a minimal or purely factual reply.\n- Answer in the format:"
            ' "Result: <YES or NO>" at the end.'
        )
        content = f"User input: {instruction}\nModel's Response: {response}"
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    prompt = f"{system_prompt}\n\n{content}"
    return prompt, system_prompt, content


def generate_eval_response(
    judge: VLLMInference,
    data: Dict[str, Any],
    remove_instruction: bool = False,
) -> tuple[List[Dict[str, str]], str]:
    prompt, system_prompt, content = build_eval_prompt_components(
        data, remove_instruction
    )
    response = judge.generate_response([prompt])[0]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return messages, response


def run_llm_evaluation(args: Any, judge: VLLMInference) -> None:
    input_response_data_path = Path(args.input_response_data)
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
            for data in tqdm(datas, desc="Generating LLM judgments"):
                messages, response = generate_eval_response(
                    judge, data, remove_instruction=remove_instruction
                )
                data["eval_response"] = response
                data["messages"] = messages
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                logging.info(json.dumps(data, ensure_ascii=False))

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
                    result = extract_result(data.get("eval_response", ""))
                    correct = bool(result and result.lower() == "yes")
                    dataset_group[dataset].append(1 if correct else 0)
                    data["correct"] = correct
                elif metric == "wer":
                    hyp = normalize_text(data.get("eval_response", ""))
                    ref = normalize_text(data.get("label", ""))
                    hyps.append(hyp)
                    refs.append(ref)
                    data["correct"] = wer(truth=[ref], hypothesis=[hyp])
                elif metric == "cot":
                    result = extract_result(data.get("eval_response", ""))
                    correct = bool(result and result.lower() == "yes")
                    dataset_group["cot"].append(1 if correct else 0)
                    data["correct"] = correct
                else:
                    logging.warning(f"Skipping unsupported metric {metric}")
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

        if refs:
            wer_score = wer(truth=refs, hypothesis=hyps)
            logging.info(f"WER: {wer_score}")
            print(f"WER: {wer_score}")
        for dataset, corrects in dataset_group.items():
            if not corrects:
                continue
            accuracy = sum(corrects) / len(corrects)
            logging.info(f"{dataset} ACC: {accuracy}")
            print(f"{dataset} ACC: {accuracy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM evaluation runner using VLLM")
    parser.add_argument(
        "--input_response_data",
        "-i",
        type=str,
        required=True,
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
        default="Qwen/Qwen3-8B",
        help="Model name passed to vLLM (default: Qwen/Qwen3-8B).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge = VLLMInference(model_name=args.judge_name)
    run_llm_evaluation(args, judge)


if __name__ == "__main__":
    main()

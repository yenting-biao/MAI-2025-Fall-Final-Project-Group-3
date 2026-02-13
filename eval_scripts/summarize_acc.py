import argparse
import json
from typing import List
from jiwer import wer
from whisper_normalizer.basic import BasicTextNormalizer
import re


def normalize_text(text: str) -> str:
    normalizer = BasicTextNormalizer()
    normalized_text = text.replace("<", "").replace(">", "")
    return normalizer(normalized_text).strip()


def load_jsonl(file_path: str) -> List[dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_all_text_after_result(text: str) -> str | None:
    pattern = r"(?i)(?<=result:\s)(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None


def parse_args():
    parser = argparse.ArgumentParser()

    # Model and Task Settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen",
        choices=[
            "qwen",
            "qwen2",
            "desta2_5",
            "blsp-emo",
            "qwen25_omni",
            "cascade_llama-3_1-8b-instruct",
            "cascade_qwen-7b-chat",
            "cascade_qwen25-7b-instruct",
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
        ],
        help="Name of the pre-trained language model to use.",
    )

    parser.add_argument(
        "--audio_task",
        type=str,
        default="ASR",
        choices=["ASR", "SER", "GR", "MMAU"],
        help="The specific audio-related task.",
    )

    parser.add_argument(
        "--response_task",
        type=str,
        default="closed_ended_questions",
        choices=[
            "closed_ended_questions",
            "chain-of-thought",
            "creative_writing",
        ],
        help="The specific task for in-context learning.",
    )

    parser.add_argument(
        "--IF_task",
        type=str,
        default=None,
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
        ],
        help="The format constraint task (i.e., instruction) for the model's response.",
    )

    parser.add_argument(
        "--no_output_constraints",
        action="store_true",
        help="Whether to summarize the model's performance without output constraints responses",
    )
    parser.add_argument(
        "--no_audio_icl",
        action="store_true",
        help="Whether to summarize the model's performance without audio ICL examples",
    )

    parser.add_argument(
        "--detail_output",
        action="store_true",
        help="Whether to print detailed performance for each IF task and each shot number. If not set, an easy to paste summary table will be printed instead.",
    )

    parser.add_argument(
        "--if_only",
        action="store_true",
        help="Whether to summarize only the model's performance on IF tasks. If set, the script will only print the IF Rate without the task performance.",
    )

    parser.add_argument(
        "--task_only",
        action="store_true",
        help="Whether to summarize only the model's performance on the main task. If set, the script will only print the task performance without the IF Rate.",
    )

    args = parser.parse_args()

    if args.if_only and args.task_only:
        raise ValueError(
            "Cannot set both --if_only and --task_only. Please choose one or neither."
        )

    return args


def main():
    args = parse_args()

    base_dir = "../model_responses"
    if args.no_output_constraints:
        base_dir += "_no_constraints"
    if args.no_audio_icl:
        base_dir += "_no_audio_icl"
    base_dir += f"/{args.model_name.lower()}/{args.audio_task}/{args.response_task}/{args.IF_task}"
    print(f"Model: {args.model_name}")
    print(f"Audio Task: {args.audio_task}")
    print(f"Response Task: {args.response_task}")
    print(f"IF Task: {args.IF_task}")
    print(f"Summarizing model performance under {base_dir}")
    print("-" * 30)

    # CoT
    if args.response_task == "chain-of-thought":
        if args.if_only:
            print("Shot | IF Rate" if args.detail_output else "IF Rate")
        elif args.task_only:
            print(
                "Shot | Task Performance" if args.detail_output else "Task Performance"
            )
        else:
            print(
                "Shot | IF Rate | Task Performance"
                if args.detail_output
                else "IFRate | Task Performance"
            )
        for i in range(9):
            if args.no_audio_icl and i == 0:
                continue
            # IF Rate
            if not args.task_only:
                if_results = load_jsonl(
                    f"{base_dir}/reports/llm_eval@output_{i}-shot.jsonl"
                )
                if_rate = sum([res["correct"] for res in if_results]) / len(if_results)

            # Task Performance
            if not args.if_only:
                task_results = load_jsonl(
                    f"{base_dir}/reports-task-level/llm_eval@output_{i}-shot.jsonl"
                )
                if args.audio_task == "ASR":
                    hyps = []
                    refs = []
                    for item in task_results:
                        result = extract_all_text_after_result(
                            item.get("eval_response", "").split("</think>")[-1].strip()
                        )
                        hyps.append(
                            normalize_text(result if result is not None else "")
                        )
                        refs.append(normalize_text(item.get("label", "").strip()))
                    task_performance = wer(refs, hyps)
                else:
                    task_performance = sum(
                        [float(res["correct"]) for res in task_results]
                    ) / len(task_results)

            # Print results
            if args.if_only:
                if args.detail_output:
                    print(f"{i}    | {if_rate:%}")
                else:
                    print(f"{if_rate:%}")
                continue
            elif args.task_only:
                if args.detail_output:
                    if args.audio_task == "ASR":
                        print(f"{i}    | {task_performance}")
                    else:
                        print(f"{i}    | {task_performance:%}")
                else:
                    if args.audio_task == "ASR":
                        print(f"{task_performance}")
                    else:
                        print(f"{task_performance:%}")
            else:
                if args.detail_output:
                    if args.audio_task == "ASR":
                        print(f"{i}    | {if_rate:%} | {task_performance}")
                    else:
                        print(f"{i}    | {if_rate:%} | {task_performance:%}")
                else:
                    if args.audio_task == "ASR":
                        print(f"{if_rate:%}\t{task_performance}")
                    else:
                        print(f"{if_rate:%}\t{task_performance:%}")
    else:
        raise NotImplementedError(
            "Only chain-of-thought response task is supported in this script."
        )


if __name__ == "__main__":
    main()

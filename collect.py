"""
Usage examples:

# Collect results of CEQ
cd <project_root>
python collect.py -ceq \
    --responses_root model_responses_no_constraints \
    --output_fn auto

# Collect results of CW
cd <project_root>
python collect.py -cw \
    --responses_root model_responses \
    --output_fn auto

# Collect results of both CEQ and CW
cd <project_root>
python collect.py -ceq -cw \
    --responses_root model_responses \
    --output_fn auto
"""

import json
import os
import argparse
from typing import Any
import numpy as np
import pandas as pd
from config import (
    MAP_MODEL_NAME,
    MAP_AUDIO_TASK,
)

GROUP_MAP = {
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
GROUP_MAP = {k.replace(':', '_'): v for k, v in GROUP_MAP.items()}

GROUP_MAP_CEQ = {
    "change_case:english_capital": "change_case",
    "change_case:english_lowercase": "change_case",
    "detectable_format:json_format": "json_format",
    "startend:quotation": "wrapping",
    "detectable_format:title": "wrapping",
    "combination:repeat_prompt": "startend",
    "startend:end_checker": "startend",
}
GROUP_MAP_CEQ = {k.replace(':', '_'): v for k, v in GROUP_MAP_CEQ.items()}

GROUP_MAP_CW = {
    "detectable_format:number_bullet_lists": "bullet_lists",
    "length_constraints:number_words": "length_constraints",
    "length_constraints:number_sentences": "length_constraints",
    "length_constraints:number_paragraphs": "length_constraints",
    "keywords:forbidden_words": "keywords",
    "keywords:existence": "keywords",
}
GROUP_MAP_CW = {k.replace(':', '_'): v for k, v in GROUP_MAP_CW.items()}

AUDIO_TASK_PERFORMANCE_METRIC = {
    "ASR": "wer",
    "SER": "answer_correct",
    "GR": "answer_correct",
    "MMAU": "answer_correct",
}

MODEL_ORDER = list(MAP_MODEL_NAME.keys())
GROUP_ORDER_CEQ = ["change_case", "startend", "wrapping", "json_format"]
GROUP_ORDER_CW = ["bullet_lists", "keywords", "length_constraints"]

def get_rule_eval_jsonl(model_name:str, audio_task:str, response_task:str, IF_task:str, data_dir:str="model_responses") -> dict[int, dict[str, Any]]:
    IF_task = IF_task.replace(":", "_")
    file_dir = os.path.join(data_dir, model_name, audio_task, response_task, IF_task, "reports")
    files = os.listdir(file_dir)
    files = [f for f in files if f.startswith("rule_eval@output_") and f.endswith(".jsonl")]
    results = {}
    for file_name in files:
        file_path = os.path.join(file_dir, file_name)
        k = file_name.split("@output_")[-1].split("-shot")[0]
        k = int(k)
        results[k] = []
        with open(file_path, 'r') as f:
            for line in f:
                results[k].append(json.loads(line))
    if len(results) != 9:
        print(f"Warning: Expected 9 shot levels, but got {len(results)} in {file_dir}")
    return results

def createDFfromRuleEvalResults(results:dict[int, dict[str, Any]], response_task:str, performance_metric:str) -> pd.DataFrame:
    d = {"IF_task": [], "shot_level": [], "n": []}
    response_task = response_task.lower()
    if response_task == "creative_writing":
        d["if_rate"] = []
    elif response_task == "closed_ended_questions":
        d["if_rate_strict"] = []
        d["if_rate_loose"] = []
        d["mean_performance"] = []
    elif response_task == "chain_of_thoughts":
        raise NotImplementedError("Chain of thoughts not implemented yet.")
    else:
        raise ValueError(f"Unknown response task: {response_task}")

    for k, v in results.items():
        for shot_level, evaluations in v.items():
            n = len(evaluations)
            d["n"].append(n)
            d["IF_task"].append(k)
            d["shot_level"].append(shot_level)

            if response_task == "creative_writing":
                follow_flags = [eval['follow_all_instructions'] for eval in evaluations]
                d["if_rate"].append(np.mean(follow_flags))
            elif response_task == "closed_ended_questions":
                performance_scores = [eval[performance_metric] for eval in evaluations] if performance_metric in evaluations[0] else None
                strict_follow_flags = [eval['strict_follow_all_instructions'] for eval in evaluations]
                loose_follow_flags = [eval['loose_follow_all_instructions'] for eval in evaluations]
                mean_performance = np.mean(performance_scores) if performance_scores is not None else None
                if_rate_strict = np.mean(strict_follow_flags)
                if_rate_loose = np.mean(loose_follow_flags)
                d["if_rate_strict"].append(if_rate_strict)
                d["if_rate_loose"].append(if_rate_loose)
                d["mean_performance"].append(mean_performance)

    return pd.DataFrame(d)

def eval(model_name:str, response_task:str, root:str="./", responses_root:str="model_responses", to_csv:bool=True):
    response_task = response_task.lower()
    d_df = {}
    for audio_task, performance_metric in AUDIO_TASK_PERFORMANCE_METRIC.items():
        if response_task == "creative_writing" and audio_task == "MMAU":
            continue
        IF_tasks = os.listdir(os.path.join(root, responses_root, model_name, audio_task, response_task))
        IF_tasks = [ft for ft in IF_tasks if ft in GROUP_MAP.keys()]
        results = {}
        for IF_task in IF_tasks:
            results[IF_task] = get_rule_eval_jsonl(
                model_name, audio_task, response_task, IF_task,
                data_dir=os.path.join(root, responses_root)
            )
        d_df[audio_task] = createDFfromRuleEvalResults(results, response_task, performance_metric)

    if to_csv:
        for audio_task, df in d_df.items():
            folder_name = "tmp"
            os.makedirs(os.path.join(root, "analysis/2026", folder_name), exist_ok=True)
            fn = os.path.join(root, "analysis/2026", folder_name, f"{model_name}_{audio_task}_{response_task}_summary.csv")
            print(f"Saving summary to {fn}")
            df.to_csv(fn, index=False)

    return d_df

def eval_ceq(args):
    response_task = "closed_ended_questions"
    d_df = {}
    for model_name in MAP_MODEL_NAME.keys():
        print(f"------------ {model_name} ------------")
        d_df[model_name] = eval(
            model_name, response_task=response_task,
            root=args.root, responses_root=args.responses_root,
            to_csv=True
        )

    group_order = []
    for v in GROUP_MAP_CEQ.values():
        if v not in group_order and v != "other":
            group_order.append(v)

    df_audio_task = {}
    if args.output_fn.lower() == 'auto':
        fn = f"summary_ceq{args.responses_root.split('model_responses')[-1]}.xlsx"
    else:
        fn = args.output_fn

    with pd.ExcelWriter(fn, engine="openpyxl") as writer:
        for audio_task in AUDIO_TASK_PERFORMANCE_METRIC.keys():
            dfs = []
            for model_name in MODEL_ORDER:
                df = d_df[model_name][audio_task].copy()

                df["model"] = model_name
                df["IF_task_group"] = df["IF_task"].map(GROUP_MAP_CEQ)

                df = df[[
                    "IF_task_group", "IF_task", "n", "model",
                    "shot_level", "if_rate_strict", "if_rate_loose", "mean_performance"
                ]]
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)

            shot_order = sorted(df_all["shot_level"].dropna().unique())
            df_all["shot_level"] = pd.Categorical(df_all["shot_level"], categories=shot_order, ordered=True)
            df_all["IF_task_group"] = pd.Categorical(df_all["IF_task_group"], categories=group_order, ordered=True)
            df_all["model"] = pd.Categorical(df_all["model"], categories=MODEL_ORDER, ordered=True)

            df_all_compare_shots = df_all.sort_values(
                by=["IF_task_group", "IF_task", "model"],
                ascending=[True, True, True],
                kind="mergesort",
            )

            df_all = df_all.sort_values(
                by=["shot_level", "IF_task_group", "IF_task", "model"],
                ascending=[True, True, True, True],
                kind="mergesort",
            )

            df_all_grouped = df_all.groupby(["shot_level", "IF_task_group", "model"], observed=False).mean(numeric_only=True).reset_index()
            df_all_compare_shots_grouped = df_all_compare_shots.groupby(["IF_task_group", "model", "shot_level"], observed=False).mean(numeric_only=True).reset_index()

            df_all.to_excel(writer, sheet_name=audio_task, index=False)
            df_all_grouped.to_excel(writer, sheet_name=f"{audio_task}_grouped", index=False)
            df_all_compare_shots.to_excel(writer, sheet_name=f"{audio_task}_compare_shots", index=False)
            df_all_compare_shots_grouped.to_excel(writer, sheet_name=f"{audio_task}_compare_shots_grouped", index=False)

            df_audio_task[audio_task] = df_all
            df_audio_task[f"{audio_task}_grouped"] = df_all_grouped
            df_audio_task[f"{audio_task}_compare_shots"] = df_all_compare_shots
            df_audio_task[f"{audio_task}_compare_shots_grouped"] = df_all_compare_shots_grouped

            print(f"Saving combined summary to {fn}")

def eval_cw(args):
    response_task ="creative_writing"
    d_df = {}
    for model_name in MAP_MODEL_NAME.keys():
        print(f"------------ {model_name} ------------")
        d_df[model_name] = eval(
            model_name, response_task=response_task,
            root=args.root, responses_root=args.responses_root,
            to_csv=True
        )

    group_order = []
    for v in GROUP_MAP_CW.values():
        if v not in group_order and v != "other":
            group_order.append(v)

    df_audio_task = {}
    if args.output_fn.lower() == 'auto':
        fn = f"summary_cw{args.responses_root.split('model_responses')[-1]}.xlsx"
    else:
        fn = args.output_fn

    with pd.ExcelWriter(fn, engine="openpyxl") as writer:
        for audio_task in AUDIO_TASK_PERFORMANCE_METRIC.keys():
            if response_task == "creative_writing" and audio_task == "MMAU":
                continue
            dfs = []
            for model_name in MODEL_ORDER:
                df = d_df[model_name][audio_task].copy()

                df["model"] = model_name
                df["IF_task_group"] = df["IF_task"].map(GROUP_MAP_CW)

                df = df[[
                    "IF_task_group", "IF_task", "n", "model",
                    "shot_level", "if_rate"
                ]]
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)

            shot_order = sorted(df_all["shot_level"].dropna().unique())
            df_all["shot_level"] = pd.Categorical(df_all["shot_level"], categories=shot_order, ordered=True)
            df_all["IF_task_group"] = pd.Categorical(df_all["IF_task_group"], categories=group_order, ordered=True)
            df_all["model"] = pd.Categorical(df_all["model"], categories=MODEL_ORDER, ordered=True)

            df_all_compare_shots = df_all.sort_values(
                by=["IF_task_group", "IF_task", "model"],
                ascending=[True, True, True],
                kind="mergesort",
            )

            df_all = df_all.sort_values(
                by=["shot_level", "IF_task_group", "IF_task", "model"],
                ascending=[True, True, True, True],
                kind="mergesort",
            )

            df_all_grouped = df_all.groupby(["shot_level", "IF_task_group", "model"], observed=False).mean(numeric_only=True).reset_index()
            df_all_compare_shots_grouped = df_all_compare_shots.groupby(["IF_task_group", "model", "shot_level"], observed=False).mean(numeric_only=True).reset_index()

            df_all.to_excel(writer, sheet_name=audio_task, index=False)
            df_all_grouped.to_excel(writer, sheet_name=f"{audio_task}_grouped", index=False)
            df_all_compare_shots.to_excel(writer, sheet_name=f"{audio_task}_compare_shots", index=False)
            df_all_compare_shots_grouped.to_excel(writer, sheet_name=f"{audio_task}_compare_shots_grouped", index=False)

            df_audio_task[audio_task] = df_all
            df_audio_task[f"{audio_task}_grouped"] = df_all_grouped
            df_audio_task[f"{audio_task}_compare_shots"] = df_all_compare_shots
            df_audio_task[f"{audio_task}_compare_shots_grouped"] = df_all_compare_shots_grouped

    print(f"Saving combined summary to {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect evaluation results and save summaries.")
    parser.add_argument("--root", type=str, default="../../", help="Root directory for the project.")
    parser.add_argument("--responses_root", type=str, default="model_responses", help="Root directory for model responses and evaluation results.")
    parser.add_argument("--eval_ceq", "-ceq", action="store_true", help="Whether to evaluate closed-ended questions.")
    parser.add_argument("--eval_cw", "-cw", action="store_true", help="Whether to evaluate creative writing.")
    parser.add_argument("--output_fn", type=str, default='auto')
    args = parser.parse_args()

    # CEQ
    if args.eval_ceq:
        eval_ceq(args)

    # CW
    if args.eval_cw:
        eval_cw(args)

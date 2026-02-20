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
import re
from typing import Any, Sequence, Tuple, List
import numpy as np
import pandas as pd
from config import MAP_MODEL_NAME, MAP_AUDIO_TASK

ROOT_RESULTS = "analysis/2026"

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

def _normalize_for_wer(text: str) -> List[str]:
    """Normalize text into a list of word tokens for corpus WER.

    - Uppercase
    - Remove non-letter characters (except apostrophe)
    - Split on whitespace
    """
    text = (text or "").upper()
    text = re.sub(r"[^A-Z']+", " ", text)
    return text.split()


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Standard Levenshtein distance on token sequences (insert/delete/substitute cost = 1)."""
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    # Two-row DP for memory efficiency.
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution / match
            )
        prev = cur
    return prev[m]


def _corpus_wer_num_den(evaluations: Sequence[dict[str, Any]]) -> Tuple[int, int]:
    """Return (total_edit_distance, total_ref_words) for corpus-level WER."""
    total_edits = 0
    total_ref_words = 0
    for item in evaluations:
        ref_text = item.get("label") or item.get("ref") or item.get("reference") or ""
        hyp_text = (
            item.get("response_for_scoring")
            or item.get("response")
            or item.get("eval_response")
            or ""
        )
        ref_tokens = _normalize_for_wer(ref_text)
        if not ref_tokens:
            continue
        hyp_tokens = _normalize_for_wer(hyp_text)
        total_ref_words += len(ref_tokens)
        total_edits += _edit_distance(ref_tokens, hyp_tokens)
    return total_edits, total_ref_words


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
    # NOTE: keep per-row numerators/denominators so downstream grouping can do weighted (micro) averages.
    d = {"IF_task": [], "shot_level": [], "n": []}
    response_task = response_task.lower()

    if response_task == "creative_writing":
        d["if_num"] = []
        d["if_den"] = []
        d["if_rate"] = []
    elif response_task == "closed_ended_questions":
        d["if_num_strict"] = []
        d["if_num_loose"] = []
        d["if_rate_strict"] = []
        d["if_rate_loose"] = []
        d["perf_num"] = []
        d["perf_den"] = []
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
                follow_flags = [bool(ev.get("follow_all_instructions", False)) for ev in evaluations]
                if_num = int(sum(follow_flags))
                if_den = int(n)
                d["if_num"].append(if_num)
                d["if_den"].append(if_den)
                d["if_rate"].append((if_num / if_den) if if_den > 0 else np.nan)
                continue

            # closed-ended questions (CEQ)
            strict_flags = [bool(ev.get("strict_follow_all_instructions", False)) for ev in evaluations]
            loose_flags = [bool(ev.get("loose_follow_all_instructions", False)) for ev in evaluations]
            if_num_strict = int(sum(strict_flags))
            if_num_loose = int(sum(loose_flags))
            den = int(n)

            d["if_num_strict"].append(if_num_strict)
            d["if_num_loose"].append(if_num_loose)
            d["if_rate_strict"].append((if_num_strict / den) if den > 0 else np.nan)
            d["if_rate_loose"].append((if_num_loose / den) if den > 0 else np.nan)

            # --- Performance metric aggregation ---
            # WER should be corpus-level: sum(edit_distance) / sum(ref_words).
            if performance_metric == "wer":
                perf_num, perf_den = _corpus_wer_num_den(evaluations)
                d["perf_num"].append(int(perf_num))
                d["perf_den"].append(int(perf_den))
                d["mean_performance"].append((perf_num / perf_den) if perf_den > 0 else np.nan)
            else:
                # Accuracy-like metrics: micro-average = sum(correct) / n.
                if n == 0 or (not evaluations) or (performance_metric not in evaluations[0]):
                    d["perf_num"].append(np.nan)
                    d["perf_den"].append(np.nan)
                    d["mean_performance"].append(np.nan)
                else:
                    vals = [float(ev.get(performance_metric, 0.0)) for ev in evaluations]
                    perf_num = float(np.sum(vals))
                    perf_den = float(n)
                    d["perf_num"].append(perf_num)
                    d["perf_den"].append(perf_den)
                    d["mean_performance"].append((perf_num / perf_den) if perf_den > 0 else np.nan)

    return pd.DataFrame(d)

def eval(model_name:str, response_task:str, root:str="./", responses_root:str="model_responses", to_csv:bool=True):
    response_task = response_task.lower()
    response_task_abbrev = {
        "closed_ended_questions": "ceq",
        "creative_writing": "cw"
    }.get(response_task, response_task)

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
            folder_name = os.path.join(root, ROOT_RESULTS, "tmp", responses_root)
            os.makedirs(folder_name, exist_ok=True)
            fn = os.path.join(folder_name, f"summary_{response_task_abbrev}_{model_name}_{audio_task}.csv")
            print(f"Saving summary to {fn}")
            df.to_csv(fn, index=False)

    return d_df

def eval_ceq(args):
    response_task = "closed_ended_questions"
    response_task_abbrev = "ceq"
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
        fn = f"./{ROOT_RESULTS}/summary_{response_task_abbrev}{args.responses_root.split('model_responses')[-1]}.xlsx"
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
                    "shot_level",
                    "if_num_strict", "if_num_loose", "perf_num", "perf_den",
                    "if_rate_strict", "if_rate_loose", "mean_performance"
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

            # --- weighted (micro) aggregation for grouped summaries ---
            df_all_grouped_tmp = (
                df_all.groupby(["shot_level", "IF_task_group", "model"], observed=False)
                .agg(
                    n=("n", "sum"),
                    if_num_strict=("if_num_strict", "sum"),
                    if_num_loose=("if_num_loose", "sum"),
                    perf_num=("perf_num", "sum"),
                    perf_den=("perf_den", "sum"),
                )
                .reset_index()
            )
            df_all_grouped_tmp["if_rate_strict"] = np.where(
                df_all_grouped_tmp["n"] > 0,
                df_all_grouped_tmp["if_num_strict"] / df_all_grouped_tmp["n"],
                np.nan,
            )
            df_all_grouped_tmp["if_rate_loose"] = np.where(
                df_all_grouped_tmp["n"] > 0,
                df_all_grouped_tmp["if_num_loose"] / df_all_grouped_tmp["n"],
                np.nan,
            )
            df_all_grouped_tmp["mean_performance"] = np.where(
                df_all_grouped_tmp["perf_den"] > 0,
                df_all_grouped_tmp["perf_num"] / df_all_grouped_tmp["perf_den"],
                np.nan,
            )
            df_all_grouped = df_all_grouped_tmp[
                ["shot_level", "IF_task_group", "model", "n", "if_rate_strict", "if_rate_loose", "mean_performance"]
            ]

            df_all_compare_shots_grouped_tmp = (
                df_all_compare_shots.groupby(["IF_task_group", "model", "shot_level"], observed=False)
                .agg(
                    n=("n", "sum"),
                    if_num_strict=("if_num_strict", "sum"),
                    if_num_loose=("if_num_loose", "sum"),
                    perf_num=("perf_num", "sum"),
                    perf_den=("perf_den", "sum"),
                )
                .reset_index()
            )
            df_all_compare_shots_grouped_tmp["if_rate_strict"] = np.where(
                df_all_compare_shots_grouped_tmp["n"] > 0,
                df_all_compare_shots_grouped_tmp["if_num_strict"] / df_all_compare_shots_grouped_tmp["n"],
                np.nan,
            )
            df_all_compare_shots_grouped_tmp["if_rate_loose"] = np.where(
                df_all_compare_shots_grouped_tmp["n"] > 0,
                df_all_compare_shots_grouped_tmp["if_num_loose"] / df_all_compare_shots_grouped_tmp["n"],
                np.nan,
            )
            df_all_compare_shots_grouped_tmp["mean_performance"] = np.where(
                df_all_compare_shots_grouped_tmp["perf_den"] > 0,
                df_all_compare_shots_grouped_tmp["perf_num"] / df_all_compare_shots_grouped_tmp["perf_den"],
                np.nan,
            )
            df_all_compare_shots_grouped = df_all_compare_shots_grouped_tmp[
                ["IF_task_group", "model", "shot_level", "n", "if_rate_strict", "if_rate_loose", "mean_performance"]
            ]

            # Export (keep the original sheet columns; drop helper numerator/denominator columns)
            df_all_export = df_all[
                ["IF_task_group", "IF_task", "n", "model", "shot_level", "if_rate_strict", "if_rate_loose", "mean_performance"]
            ]
            df_all_compare_shots_export = df_all_compare_shots[
                ["IF_task_group", "IF_task", "n", "model", "shot_level", "if_rate_strict", "if_rate_loose", "mean_performance"]
            ]

            df_all_export.to_excel(writer, sheet_name=audio_task, index=False)
            df_all_grouped.to_excel(writer, sheet_name=f"{audio_task}_grouped", index=False)
            df_all_compare_shots_export.to_excel(writer, sheet_name=f"{audio_task}_compare_shots", index=False)
            df_all_compare_shots_grouped.to_excel(writer, sheet_name=f"{audio_task}_compare_shots_grouped", index=False)

            df_audio_task[audio_task] = df_all_export
            df_audio_task[f"{audio_task}_grouped"] = df_all_grouped
            df_audio_task[f"{audio_task}_compare_shots"] = df_all_compare_shots_export
            df_audio_task[f"{audio_task}_compare_shots_grouped"] = df_all_compare_shots_grouped

            print(f"Saving combined summary to {fn}")

def eval_cw(args):
    response_task ="creative_writing"
    response_task_abbrev = "cw"
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
        fn = f"{ROOT_RESULTS}/summary_cw{args.responses_root.split('model_responses')[-1]}.xlsx"
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
                    "shot_level", "if_num", "if_den", "if_rate"
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

            # --- weighted (micro) aggregation for grouped summaries ---
            df_all_grouped_tmp = (
                df_all.groupby(["shot_level", "IF_task_group", "model"], observed=False)
                .agg(
                    n=("n", "sum"),
                    if_num=("if_num", "sum"),
                    if_den=("if_den", "sum"),
                )
                .reset_index()
            )
            df_all_grouped_tmp["if_rate"] = np.where(
                df_all_grouped_tmp["if_den"] > 0,
                df_all_grouped_tmp["if_num"] / df_all_grouped_tmp["if_den"],
                np.nan,
            )
            df_all_grouped = df_all_grouped_tmp[["shot_level", "IF_task_group", "model", "n", "if_rate"]]

            df_all_compare_shots_grouped_tmp = (
                df_all_compare_shots.groupby(["IF_task_group", "model", "shot_level"], observed=False)
                .agg(
                    n=("n", "sum"),
                    if_num=("if_num", "sum"),
                    if_den=("if_den", "sum"),
                )
                .reset_index()
            )
            df_all_compare_shots_grouped_tmp["if_rate"] = np.where(
                df_all_compare_shots_grouped_tmp["if_den"] > 0,
                df_all_compare_shots_grouped_tmp["if_num"] / df_all_compare_shots_grouped_tmp["if_den"],
                np.nan,
            )
            df_all_compare_shots_grouped = df_all_compare_shots_grouped_tmp[["IF_task_group", "model", "shot_level", "n", "if_rate"]]

            # Export (keep the original sheet columns; drop helper numerator/denominator columns)
            df_all_export = df_all[["IF_task_group", "IF_task", "n", "model", "shot_level", "if_rate"]]
            df_all_compare_shots_export = df_all_compare_shots[["IF_task_group", "IF_task", "n", "model", "shot_level", "if_rate"]]

            df_all_export.to_excel(writer, sheet_name=audio_task, index=False)
            df_all_grouped.to_excel(writer, sheet_name=f"{audio_task}_grouped", index=False)
            df_all_compare_shots_export.to_excel(writer, sheet_name=f"{audio_task}_compare_shots", index=False)
            df_all_compare_shots_grouped.to_excel(writer, sheet_name=f"{audio_task}_compare_shots_grouped", index=False)

            df_audio_task[audio_task] = df_all_export
            df_audio_task[f"{audio_task}_grouped"] = df_all_grouped
            df_audio_task[f"{audio_task}_compare_shots"] = df_all_compare_shots_export
            df_audio_task[f"{audio_task}_compare_shots_grouped"] = df_all_compare_shots_grouped

    print(f"Saving combined summary to {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect evaluation results and save summaries.")
    parser.add_argument("--root", type=str, default="./", help="Root directory for the project.")
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

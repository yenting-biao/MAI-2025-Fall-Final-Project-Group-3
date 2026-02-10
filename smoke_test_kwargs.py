#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict

def get_kwarg(d: dict, key: str):
    for kw in d.get("kwargs", []) or []:
        if key in kw:
            return kw[key]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to closed_ended_questions_corrected_filtered.jsonl")
    ap.add_argument("--show", type=int, default=10, help="How many unique values to print per task")
    ap.add_argument("--examples", type=int, default=2, help="How many example IDs to show per unique value")
    args = ap.parse_args()

    tasks = {
        "combination:repeat_prompt": "prompt_to_repeat",
        "startend:end_checker": "end_phrase",
    }

    total_by_task = Counter()
    missing_by_task = Counter()
    non_str_by_task = Counter()

    # value -> count + example ids
    value_count = {t: Counter() for t in tasks}
    value_examples = {t: defaultdict(list) for t in tasks}

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] JSON decode error at line {line_no}: {e}")

            ids = d.get("instruction_id_list") or []
            if len(ids) != 1:
                # If this ever happens, it means schema changed
                raise SystemExit(f"[ERROR] line {line_no}: instruction_id_list len != 1: {ids}")

            t = ids[0]
            if t not in tasks:
                continue

            total_by_task[t] += 1
            key = tasks[t]
            v = get_kwarg(d, key)

            if v is None or v == "":
                missing_by_task[t] += 1
                continue

            if not isinstance(v, str):
                non_str_by_task[t] += 1
                v = str(v)

            value_count[t][v] += 1
            if len(value_examples[t][v]) < args.examples:
                value_examples[t][v].append(d.get("id", "<no-id>"))

    # Report
    print("== Smoke Test Report ==")
    for t, key in tasks.items():
        tot = total_by_task[t]
        miss = missing_by_task[t]
        nstr = non_str_by_task[t]
        uniq = len(value_count[t])

        print(f"\nTask: {t}")
        print(f"  Expected kwarg key: {key}")
        print(f"  Total cases: {tot}")
        print(f"  Missing/empty {key}: {miss}")
        print(f"  Non-string {key}: {nstr}")
        print(f"  Unique {key} values: {uniq}")

        if miss > 0:
            raise SystemExit(f"\n[FAIL] Found {miss} cases missing {key} for task {t}.")

        if tot == 0:
            print("  (No cases found for this task.)")
            continue

        print(f"\n  Top {min(args.show, uniq)} values:")
        for v, c in value_count[t].most_common(args.show):
            lst = value_examples[t][v]
            lst = [str(x) for x in lst]
            ex_ids = ", ".join(lst)
            print(f"    - {v!r}  (count={c}, example_ids=[{ex_ids}])")

    print("\n[PASS] All checked tasks have the expected kwargs.")

if __name__ == "__main__":
    main()

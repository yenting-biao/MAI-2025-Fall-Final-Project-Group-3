'''
debug_context_swap.py

Usage examples:

# ASR + repeat_prompt
python debug_context_swap.py \
  --audio_task ASR \
  --if_task combination:repeat_prompt \
  --no_output_constraints \
  --examples 3 --n 2 --pick first

# SER + end_checker
python debug_context_swap.py \
  --audio_task SER \
  --if_task startend:end_checker \
  --no_output_constraints \
  --examples 3 --n 2 --pick random --seed 0

# GR + GR + repeat_prompt
python debug_context_swap.py \
  --audio_task GR \
  --if_task combination:repeat_prompt \
  --no_output_constraints \
  --examples 4 --n 1 --pick random --seed 2026

# MMAU + repeat_prompt
python debug_context_swap.py \
  --audio_task MMAU \
  --if_task combination:repeat_prompt \
  --no_output_constraints \
  --examples 3 --n 2 --pick random --seed 2026

# MMAU + end_checker
python debug_context_swap.py \
  --audio_task MMAU \
  --if_task startend:end_checker \
  --no_output_constraints \
  --examples 3 --n 2 --pick random --seed 2026
'''

import argparse
import copy
import json
import random
from pathlib import Path

import run

# --------- dataset mapping (test jsonl uses these names) ----------
DATASET_NAME_MAP = {
    "ASR": "Automatic_speech_recognition",
    "SER": "Speech_emotion_recognition",
    "GR": "Gender_recognition",
    "MMAU": "MMAU",
}

# --------- helpers ----------
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def build_conversation(icl_examples, test_case, icl_audio_dir, test_audio_dir, no_output_constraints: bool):
    """Mimic GenerateICLandTestExamples() but without model, only build the context list."""
    convo = []
    for ex in icl_examples:
        inst = ex["instruction"]
        if no_output_constraints:
            inst = run.remove_output_constraints_from_instruction(inst)
        ans = ex["ans"]
        convo.append({
            "audio_path": str(Path(icl_audio_dir) / ex["audio_path"]),
            "instruction": inst,
            "answer": json.dumps(ans, ensure_ascii=False) if isinstance(ans, dict) else str(ans),
        })

    test_inst = test_case["instruction"]
    if no_output_constraints:
        test_inst = run.remove_output_constraints_from_instruction(test_inst)

    convo.append({
        "audio_path": str(Path(test_audio_dir) / test_case["audio_filepath"]),
        "instruction": test_inst,
        # no "answer" for test query
    })
    return convo

def pick_icl_pool(InContextDataset, audio_task: str, test_case: dict, if_task: str):
    """Pick the correct ICL list for ASR/SER/GR or MMAU."""
    if audio_task != "MMAU":
        return InContextDataset[audio_task]["closed_ended_questions"][if_task], None

    # MMAU: resolve main/sub task by audio id (same as run.py)
    main_task, sub_task = run.MMAU_Get_ICL_Tasks(test_case["audio_filepath"])
    icl_pool = (
        InContextDataset["MMAU"]["closed_ended_questions"]["speech"][main_task][sub_task][if_task]
    )
    return icl_pool, (main_task, sub_task)

def apply_swap_by_kwargs(icl_examples, test_case: dict, if_task: str):
    """Apply the same swap rule as your run.py main-loop block (kwargs-driven)."""
    if if_task == "combination:repeat_prompt":
        prompt_to_repeat = run._get_kwarg(test_case, "prompt_to_repeat")
        if prompt_to_repeat is None:
            return None  # indicate missing
        for ex in icl_examples:
            ex["ans"] = run._rewrite_repeat_prompt_ans(ex["ans"], prompt_to_repeat)
        return {"prompt_to_repeat": prompt_to_repeat}

    if if_task == "startend:end_checker":
        end_phrase = run._get_kwarg(test_case, "end_phrase")
        if end_phrase is None:
            return None
        for ex in icl_examples:
            ex["ans"] = run._rewrite_end_checker_ans(ex["ans"], end_phrase)
        return {"end_phrase": end_phrase}

    raise ValueError(f"Unsupported IF task for this script: {if_task}")

def short(s: str, n=128):
    s = str(s)
    return s if len(s) <= n else s[:n] + " ..."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_task", required=True, choices=["ASR", "SER", "GR", "MMAU"])
    ap.add_argument("--if_task", required=True, choices=["combination:repeat_prompt", "startend:end_checker"])
    ap.add_argument("--test_jsonl", default="./data/eval_data/closed_ended_questions_corrected_filtered.jsonl")
    ap.add_argument("--icl_json", default="./in-context-examples/ICL_examples.json")
    ap.add_argument("--icl_audio_dir", default=".")
    ap.add_argument("--test_audio_dir", default=".")
    ap.add_argument("--examples", type=int, default=3, help="number of ICL examples")
    ap.add_argument("--n", type=int, default=3, help="number of test cases to print")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_output_constraints", action="store_true", help="mimic --no_output_constraints behavior")
    ap.add_argument("--pick", choices=["first", "random"], default="first")
    args = ap.parse_args()

    random.seed(args.seed)

    # Load ICL examples
    InContextDataset = load_json(args.icl_json)

    # Filter test cases
    dataset_name = DATASET_NAME_MAP[args.audio_task]
    candidates = []
    for d in iter_jsonl(args.test_jsonl):
        if d.get("dataset") != dataset_name:
            continue
        if d.get("instruction_id_list") != [args.if_task]:
            continue
        candidates.append(d)

    if not candidates:
        raise SystemExit(f"[ERROR] No test cases found for dataset={dataset_name}, if_task={args.if_task}")

    if args.pick == "random":
        random.shuffle(candidates)

    selected = candidates[: args.n]

    print("\n" + "=" * 90)
    print(f"audio_task = {args.audio_task} ({dataset_name})")
    print(f"if_task    = {args.if_task}")
    print(f"no_output_constraints = {args.no_output_constraints}")
    print(f"examples   = {args.examples}, n_test = {args.n}, pick = {args.pick}, seed = {args.seed}")
    print("=" * 90 + "\n")

    for idx, test_case in enumerate(selected):
        # Pick ICL pool (and MMAU main/sub if needed)
        icl_pool, mmau_task = pick_icl_pool(InContextDataset, args.audio_task, test_case, args.if_task)
        if len(icl_pool) == 0:
            print(f"[WARN] empty icl_pool for test_id={test_case.get('id')}, skip")
            continue

        # Sample ICL examples
        pool = list(icl_pool)
        random.shuffle(pool)
        icl_sample = pool[: min(args.examples, len(pool))]

        # Prepare no-swap and swap copies
        icl_no = copy.deepcopy(icl_sample)
        icl_yes = copy.deepcopy(icl_sample)

        swap_info = apply_swap_by_kwargs(icl_yes, test_case, args.if_task)

        # Build conversations (context lists)
        convo_no = build_conversation(icl_no, test_case, args.icl_audio_dir, args.test_audio_dir, args.no_output_constraints)
        convo_yes = build_conversation(icl_yes, test_case, args.icl_audio_dir, args.test_audio_dir, args.no_output_constraints)

        # Header
        print("\n" + "-" * 90)
        print(f"[{idx+1}/{len(selected)}] test_id={test_case.get('id')}  audio={test_case.get('audio_filepath')}")
        if mmau_task is not None:
            print(f"  MMAU task: main={mmau_task[0]} | sub={mmau_task[1]}")
        print(f"  kwargs used: \033[93m{swap_info}\033[0m")
        print("-" * 90)

        # Show ICL answer diffs
        print("\n[ICL ANSWER DIFFS] (no-swap  vs  swap)")
        for i, (a, b) in enumerate(zip(icl_no, icl_yes), 1):
            a_ans = a["ans"]
            b_ans = b["ans"]
            changed = (str(a_ans) != str(b_ans))

            print(f"\n  ICL #{i} changed={changed}")
            print(f"    instruction (raw first line): {short(a['instruction'].splitlines()[0])}")
            if args.no_output_constraints:
                inst_stripped = run.remove_output_constraints_from_instruction(a["instruction"])
                print(f"    instruction (stripped):      {short(inst_stripped)}")

            print(f"    ans (no-swap): {short(a_ans, 256)}")
            print(f"    ans (swap):    {short(b_ans, 256)}")

        # Print full context JSON (what would be fed into model.process_input)
        print("\n[CONTEXT JSON | NO-SWAP]")
        print(json.dumps(convo_no, ensure_ascii=False, indent=2))

        print("\n[CONTEXT JSON | SWAP]")
        print(json.dumps(convo_yes, ensure_ascii=False, indent=2))

        # Simple sanity checks
        if swap_info is None:
            print("\n[WARN] kwargs missing for this test case; swap did not happen.")
        else:
            if args.if_task == "combination:repeat_prompt":
                pref = swap_info["prompt_to_repeat"].strip()
                ok = all(str(ex["ans"]).lstrip().startswith(pref) for ex in icl_yes)
                print(f"\n[CHECK] all swapped ICL answers start with prompt_to_repeat?  {ok}")
            elif args.if_task == "startend:end_checker":
                suf = swap_info["end_phrase"].strip()
                ok = all(str(ex["ans"]).rstrip().endswith(suf) for ex in icl_yes)
                print(f"\n[CHECK] all swapped ICL answers end with end_phrase?  {ok}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

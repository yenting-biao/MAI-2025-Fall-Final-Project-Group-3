'''
debug_context_swap.py

Usage examples:

# ASR + repeat_prompt
python debug_context_swap.py \
  --audio_task ASR \
  --IF_task combination:repeat_prompt \
  --no_output_constraints \
  --examples 3 --n 2 --pick first

# SER + end_checker
python debug_context_swap.py \
  --audio_task SER \
  --IF_task startend:end_checker \
  --no_output_constraints \
  --examples 3 --n 2 --pick random --seed 0

# GR + GR + repeat_prompt
python debug_context_swap.py \
  --audio_task GR \
  --IF_task combination:repeat_prompt \
  --no_output_constraints \
  --examples 4 --n 1 --pick random --seed 2026

# MMAU + repeat_prompt
python debug_context_swap.py \
  --audio_task MMAU \
  --IF_task combination:repeat_prompt \
  --no_output_constraints \
  --examples 3 --n 2 --pick random --seed 2026

# MMAU + end_checker
python debug_context_swap.py \
  --audio_task MMAU \
  --IF_task startend:end_checker \
  --no_output_constraints \
  --examples 3 --n 2 --pick random --seed 2026
'''

import argparse
import copy
import json
import random
from typing import Any

import run


def _short(x: Any, n: int = 220) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ..."


def _strip_instruction_or_die(instr: str) -> str:
    """Match run.py behavior; fail loudly if instruction format is unexpected."""
    try:
        return run.remove_output_constraints_from_instruction(instr)
    except Exception as e:
        print("\n[ERROR] remove_output_constraints_from_instruction failed.")
        print(f"  instruction newlines = {instr.count(chr(10))}")
        print("  instruction preview:")
        print(_short(instr, 800))
        raise


def _build_test_case_formatted(test_audio_dir: str, test_case: dict, use_test_sample: bool) -> dict:
    if use_test_sample:
        return test_case
    return {
        "audio_path": run.os.path.join(test_audio_dir, test_case["audio_filepath"]),
        "instruction": test_case["instruction"],
    }


def _select_icl_pool(args_ns, icl_data, test_case: dict):
    """
    Mirror run.main():
      - MMAU: icl_data[main_task][sub_task][IF_task]
      - else: icl_data (list)
    """
    if args_ns.audio_task == "MMAU":
        main_task, sub_task = run.MMAU_Get_ICL_Tasks(test_case["audio_filepath"])
        pool = icl_data[main_task][sub_task][args_ns.IF_task].copy() if args_ns.examples > 0 else []
        return pool, (main_task, sub_task)
    else:
        pool = icl_data.copy() if args_ns.examples > 0 else []
        return pool, None


def _maybe_strip_test_instruction(args_ns, test_case: dict):
    """Match run.main(): strip test_case['instruction'] when --no_output_constraints."""
    if args_ns.no_output_constraints:
        test_case["instruction"] = _strip_instruction_or_die(test_case["instruction"])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--audio_task", required=True, choices=["ASR", "SER", "GR", "MMAU"])
    ap.add_argument("--IF_task", required=True, choices=["combination:repeat_prompt", "startend:end_checker"])

    ap.add_argument("--response_task", default="closed_ended_questions",
                    choices=["closed_ended_questions", "creative_writing", "chain-of-thought"])
    ap.add_argument("--examples", type=int, default=3, help="number of ICL examples (k-shot)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=3, help="number of test cases to print")
    ap.add_argument("--pick", choices=["first", "random"], default="first")

    # match run.parse_args defaults
    ap.add_argument("--icl_json_path", default="./in-context-examples/ICL_examples.json")
    ap.add_argument("--icl_audio_dir", default="./in-context-examples/audios/")
    ap.add_argument("--test_audio_dir", default="./data/audios/")
    ap.add_argument("--test_eval_dir", default="./data/eval_data/")
    ap.add_argument("--use_test_sample", action="store_true")

    # IMPORTANT: you said you must use this in experiments this round
    ap.add_argument("--no_output_constraints", action="store_true", default=True)
    ap.add_argument("--no_output_constraints_off", action="store_true",
                    help="If set, disables no_output_constraints (for debugging only).")

    ap.add_argument("--print_full_context", action="store_true",
                    help="Print the entire conversation JSON (can be long).")
    args = ap.parse_args()

    if args.no_output_constraints_off:
        args.no_output_constraints = False

    # Build a Namespace compatible with run.GetICLData / run.GetTestCases
    # (run.py expects these fields)
    args_ns = argparse.Namespace(
        model_name="qwen",               # not used here (we don't load model)
        device="cpu",                    # not used here
        seed=args.seed,
        verbose=False,
        debug=False,
        debug_examples=10,

        output_dir="./model_responses/",

        icl_json_path=args.icl_json_path,
        icl_audio_dir=args.icl_audio_dir,

        no_output_constraints=args.no_output_constraints,

        test_audio_dir=args.test_audio_dir,
        test_eval_dir=args.test_eval_dir,
        use_test_sample=args.use_test_sample,

        audio_task=args.audio_task,
        response_task=args.response_task,
        IF_task=args.IF_task,

        examples=args.examples,
    )

    random.seed(args_ns.seed)

    audio_task_mapped = run.MAP_AUDIO_TASK[args_ns.audio_task.upper()]  # from config via run.py

    # Load ICL data and test cases using run.py code paths
    icl_data = run.GetICLData(args_ns) if args_ns.examples > 0 else []
    test_cases, test_audio_dir = run.GetTestCases(args_ns, audio_task_mapped)

    if not test_cases:
        raise SystemExit(
            f"[ERROR] No test cases found. "
            f"audio_task_mapped={audio_task_mapped}, response_task={args_ns.response_task}, IF_task={args_ns.IF_task}\n"
            f"Check --test_eval_dir and that the jsonl contains matching entries."
        )

    if args.pick == "random":
        random.shuffle(test_cases)

    selected = test_cases[:args.n]

    print("\n" + "=" * 100)
    print(f"audio_task         = {args_ns.audio_task} ({audio_task_mapped})")
    print(f"response_task      = {args_ns.response_task}")
    print(f"IF_task            = {args_ns.IF_task}")
    print(f"examples (k-shot)  = {args_ns.examples}")
    print(f"no_output_constraints = {args_ns.no_output_constraints}")
    print(f"n_test_cases       = {len(selected)} (pick={args.pick}, seed={args_ns.seed})")
    print("=" * 100 + "\n")

    for idx, tc in enumerate(selected, 1):
        # Work on copies so we don't mutate test_cases list
        test_case_no = copy.deepcopy(tc)
        test_case_sw = copy.deepcopy(tc)

        # Match run.main(): strip test instruction when no_output_constraints 
        _maybe_strip_test_instruction(args_ns, test_case_sw)

        # Select ICL pool exactly like run.main()
        pool, mmau_task = _select_icl_pool(args_ns, icl_data, test_case_no)
        if len(pool) == 0:
            print(f"[WARN] Empty ICL pool for test_id={test_case_no.get('id')}, skip.")
            continue

        random.shuffle(pool)
        icl_no = copy.deepcopy(pool[:args_ns.examples])
        icl_sw = copy.deepcopy(pool[:args_ns.examples])

        # Apply rewrite ONLY for swap version, exactly as run.main() does
        if args_ns.no_output_constraints and args_ns.examples > 0:
            icl_sw = run.rewrite_ans(args_ns, test_case_sw, icl_sw)

        # Build conversations using run.GenerateICLandTestExamples
        test_case_formatted_no = _build_test_case_formatted(test_audio_dir, test_case_no, args_ns.use_test_sample)
        test_case_formatted_sw = _build_test_case_formatted(test_audio_dir, test_case_sw, args_ns.use_test_sample)

        convo_no = run.GenerateICLandTestExamples(
            icl_no, args_ns.icl_audio_dir, test_case_formatted_no,
            debug=False, remove_output_constraints=args_ns.no_output_constraints
        )
        convo_sw = run.GenerateICLandTestExamples(
            icl_sw, args_ns.icl_audio_dir, test_case_formatted_sw,
            debug=False, remove_output_constraints=args_ns.no_output_constraints
        )

        # Print header
        print("\n" + "-" * 100)
        print(f"[{idx}/{len(selected)}] test_id={test_case_no.get('id')} audio={test_case_no.get('audio_filepath')}")
        if mmau_task:
            print(f"  MMAU resolved: main_task={mmau_task[0]} | sub_task={mmau_task[1]}")
        print(f"  kwargs: \033[93m{_short(test_case_no.get('kwargs', []), 500)}\033[0m")
        if args_ns.IF_task == "combination:repeat_prompt":
            print(f"  prompt_to_repeat = {run._get_kwarg(test_case_no, 'prompt_to_repeat')!r}")
        if args_ns.IF_task == "startend:end_checker":
            print(f"  end_phrase       = {run._get_kwarg(test_case_no, 'end_phrase')!r}")
        print("-" * 100)

        # Print per-ICL answer diff (what you care about)
        print("\n[ICL ANSWER DIFFS] (no-swap vs swap)")
        for i_ex, (ex0, ex1) in enumerate(zip(icl_no, icl_sw), 1):
            a0 = ex0.get("ans")
            a1 = ex1.get("ans")
            changed = str(a0) != str(a1)

            print(f"\n  ICL #{i_ex} changed={changed}")
            print(f"    ICL audio_path: {ex0.get('audio_path')}")
            print(f"    instr first line (raw): {_short(ex0.get('instruction', '').splitlines()[0], 180)}")
            if args_ns.no_output_constraints:
                stripped = _strip_instruction_or_die(ex0.get("instruction", ""))
                print(f"    instr (stripped):      {_short(stripped, 180)}")
            print(f"    ans (no-swap): {_short(a0, 300)}")
            print(f"    ans (swap):    {_short(a1, 300)}")

        # Print conversation JSON (context message)
        print("\n[CONTEXT | NO-SWAP] (conversation list that will be fed into model.process_input)")
        if args.print_full_context:
            print(json.dumps(convo_no, ensure_ascii=False, indent=2))
        else:
            print(_short(convo_no, 1200))

        print("\n[CONTEXT | SWAP] (conversation list that will be fed into model.process_input)")
        if args.print_full_context:
            print(json.dumps(convo_sw, ensure_ascii=False, indent=2))
        else:
            print(_short(convo_sw, 1200))

    print("\nDone.\n")


if __name__ == "__main__":
    main()

import json


def check(rephrased: set, original: set):
    # rephrased len 50 and all different from original
    assert (
        len(rephrased) == 50 and len(rephrased.intersection(original)) == 0
    ), f"Rephrased prompts do not meet the criteria: len={len(rephrased)}, overlap={len(rephrased.intersection(original))}"


rephrased_data = json.load(open(f"./rephrased_prompt_types.json"))
orig_data = json.load(open(f"./Speech-IFEval_prompt_types.json"))

for type_key in rephrased_data:
    rephrased_prompts = set(rephrased_data[type_key])
    orig_prompts = set(orig_data[type_key])
    check(rephrased_prompts, orig_prompts)
    print(f"{type_key} prompts rephrased correctly: {len(rephrased_prompts)} prompts.")

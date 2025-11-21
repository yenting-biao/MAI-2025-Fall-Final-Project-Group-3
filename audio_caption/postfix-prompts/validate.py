import json

paths = [
    "chain-of-thoughts.json",
    "closed_ended_questions.json",
    "creative_writing.json",
]


def check(rephrased: set, original: set):
    # rephrased len 25 and all different from original
    assert (
        len(rephrased) == 25 and len(rephrased.intersection(original)) == 0
    ), f"Rephrased prompts do not meet the criteria: len={len(rephrased)}, overlap={len(rephrased.intersection(original))}"


for path in paths:
    rephrased_data = json.load(open(f"./rephrased-prompts/{path}"))
    orig_data = json.load(open(f"./Speech-IFEval-prompts/{path}"))

    if path == "chain-of-thoughts.json":
        rephrased_prompts = set(rephrased_data)
        orig_prompts = set(orig_data)
        check(rephrased_prompts, orig_prompts)
        print(f"CoT prompts rephrased correctly: {len(rephrased_prompts)} prompts.")
    else:
        for key in rephrased_data:
            rephrased_prompts = set(rephrased_data[key])
            orig_prompts = set(orig_data[key])
            check(rephrased_prompts, orig_prompts)
            print(
                f"{path} - {key} prompts rephrased correctly: {len(rephrased_prompts)} prompts."
            )

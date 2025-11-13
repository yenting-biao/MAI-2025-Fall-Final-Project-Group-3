import json
from typing import List
import random

random.seed(42)


def merge_prompts(prefix: List[dict], postfix: List[str]) -> List[dict]:
    assert (
        len(prefix) == len(postfix) * 2
    ), "Prefix and postfix prompt counts do not match."
    random.shuffle(postfix)
    random.shuffle(prefix)

    merged = []
    for i in range(len(prefix)):
        prompt = prefix[i]["instruction"] + "\n" + postfix[i % len(postfix)]
        merged.append(
            {
                "audio_path": (
                    prefix[i]["audio_path"]
                    if prefix[i]["audio_path"] != ""
                    else f"general/{i}.wav"
                ),
                "instruction": prompt,
                "ans": prefix[i]["ans"],
            }
        )
    return merged


def load_task_type_prompts():
    data = {
        "ASR": [],
        "SER": [],
        "GR": [],
        "MMAU": [],
    }

    with open("../audio_caption/prefix-prompts/rephrased_prompt_types.json") as f:
        prompt_types = json.load(f)
    for type_key in prompt_types.keys():
        # data[type_key] = prompt_types[type_key]
        for i, prompt in enumerate(prompt_types[type_key]):
            data[type_key].append(
                {
                    "audio_path": "",
                    "instruction": prompt,
                    "ans": "",
                }
            )

    with open("../audio_caption/prefix-prompts/sampled_mmau-test-mini.json") as f:
        mmau_data = json.load(f)
    data["MMAU"] = mmau_data

    return data


def load_postfix_constraints():
    data = {
        "chain-of-thought": [],
        "closed_ended_questions": {},
        "creative_writing": {},
    }

    with open(
        "../audio_caption/postfix-prompts/rephrased-prompts/chain-of-thoughts.json"
    ) as f:
        cot_prompts = json.load(f)
    data["chain-of-thought"] = cot_prompts

    with open(
        "../audio_caption/postfix-prompts/rephrased-prompts/closed_ended_questions.json"
    ) as f:
        ceq_prompts = json.load(f)
    data["closed_ended_questions"] = ceq_prompts

    with open(
        "../audio_caption/postfix-prompts/rephrased-prompts/creative_writing.json"
    ) as f:
        cw_prompts = json.load(f)
    data["creative_writing"] = cw_prompts

    return data


def main():
    task_type_prompts = load_task_type_prompts()
    postfix_constraints = load_postfix_constraints()
    merged_outputs = {task_type: {} for task_type in task_type_prompts.keys()}
    for task_type in task_type_prompts.keys():
        for constraint_type in postfix_constraints.keys():
            if constraint_type == "chain-of-thought":
                merged_outputs[task_type][constraint_type] = merge_prompts(
                    task_type_prompts[task_type], postfix_constraints[constraint_type]
                )
            elif constraint_type == "creative_writing":
                if constraint_type not in merged_outputs[task_type]:
                    merged_outputs[task_type][constraint_type] = {}
                for format_constraint in postfix_constraints[constraint_type].keys():
                    merged_outputs[task_type][constraint_type][format_constraint] = []
                    for i, prompt in enumerate(
                        postfix_constraints[constraint_type][format_constraint]
                    ):
                        output = {
                            "audio_path": (
                                f"general/{i}.wav"
                                if task_type != "MMAU"
                                else task_type_prompts[task_type][i]["audio_path"]
                            ),
                            "instruction": prompt,
                            "ans": "",
                        }
                        merged_outputs[task_type][constraint_type][
                            format_constraint
                        ].append(output)
                # if constraint_type not in merged_outputs[task_type]:
                # merged_outputs[task_type][constraint_type] = {}
                # merged_outputs[task_type][constraint_type] = postfix_constraints[
                #     constraint_type
                # ]
            elif constraint_type == "closed_ended_questions":
                for format_constraint in postfix_constraints[constraint_type].keys():
                    if constraint_type not in merged_outputs[task_type]:
                        merged_outputs[task_type][constraint_type] = {}
                    merged_outputs[task_type][constraint_type][format_constraint] = (
                        merge_prompts(
                            task_type_prompts[task_type],
                            postfix_constraints[constraint_type][format_constraint],
                        )
                    )
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")

    with open("generated_in_context_examples.json", "w") as f:
        json.dump(merged_outputs, f, indent=4, ensure_ascii=False)
    print("Generated in-context examples saved to generated_in_context_examples.json")


if __name__ == "__main__":
    main()

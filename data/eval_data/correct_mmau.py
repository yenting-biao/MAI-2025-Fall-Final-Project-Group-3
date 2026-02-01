import json
from argparse import ArgumentParser

input_files = ["chain-of-thought.jsonl", "closed_ended_questions.jsonl"]
mmau_test_mini_path = "../../../MMAU/mmau-test-mini.json"
mmau_data = json.load(open(mmau_test_mini_path))
mmau_data = {item["id"]: item for item in mmau_data}


def option_to_string(options: list[str]) -> str:
    # "optionA", "optionB", ..., "optionN-1" or "optionN"
    if len(options) == 1:
        return f'"{options[0]}"'
    else:
        return ", ".join(f'"{opt}"' for opt in options[:-1]) + f' or "{options[-1]}"'


def process_data(input_data: list[dict]) -> tuple[int, int, list[dict]]:
    error_count = 0
    mmau_count = 0
    for ifeval_item in input_data:
        dataset = ifeval_item["dataset"]
        if dataset != "MMAU":
            continue

        mmau_count += 1
        mmau_id = ifeval_item["audio_filepath"].split("/")[-1].replace(".wav", "")
        mmau_item = mmau_data[mmau_id]

        mmau_question = (
            mmau_item["question"]
            + " Choose the answer from "
            + option_to_string(mmau_item["choices"])
            + "."
        )
        mmau_answer = mmau_item["answer"]
        ifeval_question = ifeval_item["instruction"].split("\n")[0]
        ifeval_if_command = (
            ifeval_item["instruction"].replace(ifeval_question, "").strip()
        )
        ifeval_answer = ifeval_item["label"]

        if ifeval_question != mmau_question:
            error_count += 1
            wrong_info = {
                "MMAU_question": mmau_question,
                "MMAU_label": mmau_answer,
                "Speech-IFEval_question": ifeval_question,
                "Speech-IFEval_label": ifeval_answer,
            }
            assert mmau_answer == ifeval_answer, "Labels do not match!"
            print(f"ERROR in {mmau_id}:")
            print(json.dumps(wrong_info, indent=2))
            print()

            # correct the question
            ifeval_item["instruction"] = mmau_question + "\n" + ifeval_if_command
            ifeval_item["label"] = mmau_answer

            print("Corrected to:")
            print(json.dumps(ifeval_item, indent=2))
            print("----------------------")

    return error_count, mmau_count, input_data


if __name__ == "__main__":
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = [json.loads(line) for line in f]

        error_count, mmau_count, processed_data = process_data(input_data)
        print(
            f"Processed {mmau_count} MMAU items in {input_file}, found and corrected {error_count} errors."
        )

        output_file = input_file.replace(".jsonl", "_corrected.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in processed_data:
                f.write(json.dumps(item) + "\n")

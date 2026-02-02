"""Remove MMAU phoneme-counting and emotion-flipping questions from
Speech-IFEval dataset"""

import json
from pathlib import Path
from typing import Any

icl_path = Path("in-context-examples")
data_path = Path("data", "eval_data")

with open(icl_path / "mmau-mini-id2task.json", "r", encoding="utf-8") as f:
    test_mini_id2task: dict[str, Any] = json.load(f)
with open(icl_path / "mmau-id2task.json", "r", encoding="utf-8") as f:
    test_id2task: dict[str, Any] = json.load(f)


def filter_jsonl(speechifeval_jsonl_filename: str) -> None:
    """Remove lines containing phoneme-counting and emotion-flipping questions
    and saves the remaining lines to a new JSON Lines file"""

    lines_removed = 0
    with open(data_path / speechifeval_jsonl_filename, "r", encoding="utf-8") as f:
        new_jsonl_lines: list[str] = []
        for line in f:
            speechifeval_question: dict[str, Any] = json.loads(line)
            taskname, audio_path = speechifeval_question["audio_filepath"].split("/")
            if taskname == "MMAU":
                instruction = speechifeval_question["instruction"]
                mmau_id: str = Path(audio_path).stem
                assert (
                    mmau_id in test_mini_id2task
                    or mmau_id in test_id2task
                    and not (mmau_id in test_mini_id2task and mmau_id in test_id2task)
                )
                id2task = (
                    test_mini_id2task if mmau_id in test_mini_id2task else test_id2task
                )
                filter_out: bool = (
                    id2task[mmau_id]["sub-category"]
                    == "Phonemic Stress Pattern Analysis"
                    and "a pair of words" not in instruction
                ) or id2task[mmau_id]["sub-category"] == "Emotion Flip Detection"
                if filter_out:
                    lines_removed += 1
                    continue
            new_jsonl_lines.append(line)

    new_jsonl_filename = Path(speechifeval_jsonl_filename).stem + "_filtered.jsonl"
    with open(data_path / new_jsonl_filename, "w", encoding="utf-8") as new_file:
        for line in new_jsonl_lines:
            new_file.write(f"{line}")

    print(f"Removed {lines_removed} from {speechifeval_jsonl_filename}")


if __name__ == "__main__":
    # filter_jsonl("chain-of-thought.jsonl")
    filter_jsonl("chain-of-thought_corrected.jsonl")
    # filter_jsonl("closed_ended_questions.jsonl")
    filter_jsonl("closed_ended_questions_corrected.jsonl")

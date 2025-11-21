import json
import pandas as pd
from argparse import ArgumentParser
from typing import List


def fill_in_answer(data: List[dict], MAP: dict, format_func: callable) -> dict:
    for item in data:
        filename = item["audio_path"].split("/")[-1]
        item["ans"] = format_func(MAP[filename], item)
    return data


def load_crema_d_metadata(path: str) -> tuple[dict, dict, dict]:
    metadata_df = pd.read_csv(path)
    transcription_map = dict(zip(metadata_df["filename"], metadata_df["transcription"]))
    emotion_map = dict(zip(metadata_df["filename"], metadata_df["emotion"]))
    gender_map = dict(zip(metadata_df["filename"], metadata_df["gender"]))
    return transcription_map, emotion_map, gender_map


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="./generated_in_context_examples.json",
        help="Path to the input JSON file containing generated examples.",
    )
    parser.add_argument(
        "--metadata_file",
        "-m",
        type=str,
        default="./audios/CREMA-D_metadata.csv",
        help="Path to the CREMA-D metadata CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load generated examples
    with open(args.input_file, "r") as f:
        example_data = json.load(f)

    TRANSCRIPTION_MAP, EMOTION_MAP, GENDER_MAP = load_crema_d_metadata(
        args.metadata_file
    )

    # ASR
    example_data["ASR"]["closed_ended_questions"]["change_case:english_capital"] = (
        fill_in_answer(
            example_data["ASR"]["closed_ended_questions"][
                "change_case:english_capital"
            ],
            TRANSCRIPTION_MAP,
            lambda x, _: x.upper(),
        )
    )
    example_data["ASR"]["closed_ended_questions"]["change_case:english_lowercase"] = (
        fill_in_answer(
            example_data["ASR"]["closed_ended_questions"][
                "change_case:english_lowercase"
            ],
            TRANSCRIPTION_MAP,
            lambda x, _: x.lower(),
        )
    )

    # SER
    example_data["SER"]["closed_ended_questions"]["change_case:english_capital"] = (
        fill_in_answer(
            example_data["SER"]["closed_ended_questions"][
                "change_case:english_capital"
            ],
            EMOTION_MAP,
            lambda x, _: x.upper(),
        )
    )
    example_data["SER"]["closed_ended_questions"]["change_case:english_lowercase"] = (
        fill_in_answer(
            example_data["SER"]["closed_ended_questions"][
                "change_case:english_lowercase"
            ],
            EMOTION_MAP,
            lambda x, _: x.lower(),
        )
    )

    # GR
    wording_map = {"Male": "Man", "Female": "Woman"}

    example_data["GR"]["closed_ended_questions"]["change_case:english_capital"] = (
        fill_in_answer(
            example_data["GR"]["closed_ended_questions"]["change_case:english_capital"],
            GENDER_MAP,
            lambda x, item: (
                x.upper()
                if "Male" in item["instruction"]
                else (
                    wording_map[x].upper()
                    if "Man" in item["instruction"]
                    else "UNKNOWN"
                )
            ),
        ),
    )

    example_data["GR"]["closed_ended_questions"]["change_case:english_lowercase"] = (
        fill_in_answer(
            example_data["GR"]["closed_ended_questions"][
                "change_case:english_lowercase"
            ],
            GENDER_MAP,
            lambda x, item: (
                x.lower()
                if "Male" in item["instruction"]
                else (
                    wording_map[x].lower()
                    if "Man" in item["instruction"]
                    else "UNKNOWN"
                )
            ),
        )
    )

    # Save the updated examples with filled-in answers
    with open(args.input_file, "w") as f:
        json.dump(example_data, f, indent=4, ensure_ascii=False)

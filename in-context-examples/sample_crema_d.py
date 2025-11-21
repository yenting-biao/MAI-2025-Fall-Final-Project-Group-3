import os
import random
import shutil
from argparse import ArgumentParser
import pandas as pd
import itertools


TRANSCRIPTION_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# Two emotions are excluded since our prompt only covers neutral, happy, sad, angry
EMOTION_MAP = {
    "ANG": "Angry",
    # "DIS": "Disgust",
    # "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

GENDER_MAP = {}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source folder containing CREMA-D wav files.",
    )
    parser.add_argument(
        "--src_metadata",
        type=str,
        required=True,
        help="Path to the CREMA-D metadata CSV file.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="./audios/general",
        help="Destination folder to copy selected wav files.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=50,
        help="Number of random wav files to select and copy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_metadata_path",
        type=str,
        default="./audios/CREMA-D_metadata.csv",
        help="Path to save the metadata CSV file.",
    )
    return parser.parse_args()


def get_speaker_gender_metadata(path: str) -> dict:
    df = pd.read_csv(path)  # ActorID, Age, Sex, Race, Ethnicity
    return dict(zip(df["ActorID"], df["Sex"]))


def copy_a_file(
    i: int, src_path: str, dst_path: str, selected_file: str, metadata: list
):
    output_file = f"{i}.wav"
    shutil.copy(src_path, dst_path)
    metadata.append(
        {
            "filename": output_file,
            "gender": GENDER_MAP.get(int(selected_file.split("_")[0]), "Unknown"),
            "emotion": EMOTION_MAP.get(selected_file.split("_")[2], "Unknown"),
            "transcription": TRANSCRIPTION_MAP.get(
                selected_file.split("_")[1], "Unknown"
            ),
        }
    )


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    GENDER_MAP = get_speaker_gender_metadata(args.src_metadata)

    os.makedirs(args.dst, exist_ok=True)

    wav_files = [f for f in os.listdir(args.src) if f.lower().endswith(".wav")]
    if len(wav_files) < args.num_files:
        raise ValueError(
            f"Found only {len(wav_files)} wav files, need at least {args.num_files}."
        )
    # qualified_wav_files = [
    #     f for f in wav_files if f.split("_")[2] in EMOTION_MAP.keys()
    # ]
    # selected = random.sample(qualified_wav_files, args.num_files)
    # pairs = list(itertools.product(TRANSCRIPTION_MAP.keys(), EMOTION_MAP.keys()))
    # random.shuffle(pairs)

    transcription_counter = {key: 0 for key in TRANSCRIPTION_MAP.keys()}

    def get_least_used_transcription():
        min_count = min(transcription_counter.values())
        least_used = [k for k, v in transcription_counter.items() if v == min_count]
        return random.choice(least_used)

    metadata = []
    i = 0
    while i < args.num_files:
        for emotion_key in EMOTION_MAP.keys():
            for gender_key in ["Male", "Female"]:
                if i >= args.num_files:
                    break

                transcription_key = get_least_used_transcription()

                candidates = [
                    f
                    for f in wav_files
                    if f.split("_")[1] == transcription_key
                    and f.split("_")[2] == emotion_key
                    and GENDER_MAP[int(f.split("_")[0])] == gender_key
                ]
                if candidates:
                    selected_file = random.choice(candidates)
                    wav_files.remove(selected_file)  # Avoid re-selection
                    copy_a_file(
                        i,
                        os.path.join(args.src, selected_file),
                        os.path.join(args.dst, f"{i}.wav"),
                        selected_file,
                        metadata,
                    )
                    transcription_counter[transcription_key] += 1
                    i += 1

    df = pd.DataFrame(metadata)
    df.to_csv(args.output_metadata_path, index=False)

    print(f"Done. Copied {args.num_files} random wav files.")

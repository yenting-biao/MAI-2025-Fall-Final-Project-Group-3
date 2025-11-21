import os
import random
import shutil
from argparse import ArgumentParser
import pandas as pd
from scipy.fftpack import dst

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

EMOTION_MAP = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}


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
        default=42,
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
    selected = random.sample(wav_files, args.num_files)

    metadata = []
    for i, f in enumerate(selected):
        output_file = f"{i}.wav"
        shutil.copy(os.path.join(args.src, f), os.path.join(args.dst, output_file))
        metadata.append(
            {
                "filename": output_file,
                "gender": GENDER_MAP.get(int(f.split("_")[0]), "Unknown"),
                "emotion": EMOTION_MAP.get(f.split("_")[2], "Unknown"),
                "transcription": TRANSCRIPTION_MAP.get(f.split("_")[1], "Unknown"),
            }
        )

    df = pd.DataFrame(metadata)
    df.to_csv(args.output_metadata_path, index=False)

    print(f"Done. Copied {args.num_files} random wav files.")

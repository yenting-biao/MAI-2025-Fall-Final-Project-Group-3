import time
from typing import List, Dict, Any

from models.blsp_emo import BLSP_Emo

def load_model(
    model_cls,
    name: str,
    verbose: bool = True,
    model_kwargs: Dict[str, Any] = {},
):
    if verbose:
        print(f"[{name}] Initializing model ...", end=" ")
    t0 = time.time()
    model = model_cls(**model_kwargs)
    t1 = time.time()
    if verbose:
        print(f"\033[32mDone! Model initialized in {t1 - t0:.2f} seconds.\033[0m")

    return model


def run_smoke_test(
    model,
    name: str,
    conversation: List[Dict[str, str]],
    verbose: bool = True,
):

    print(f"=== Testing {name} ===")
    t1 = time.time()

    # --- Process Input ---
    if verbose:
        print(f"[{name}] Processing input ...", end=" ")
    model.process_input(conversation)
    t2 = time.time()
    if verbose:
        print(f"\033[32mDone! Input processed in {t2 - t1:.2f} seconds.\033[0m")

    if verbose:
        print(f"[{name}] Generating output ...", end=" ")
    out = model.generate()
    t3 = time.time()
    if verbose:
        print(f"\033[32mDone! Output generated in {t3 - t2:.2f} seconds.\033[0m")

    print(f"\n[{name}] Output:", f"\033[33m{out}\033[0m")


if __name__ == "__main__":
    # Warning: at the moment, same as DiVA, the inputted system prompt is not used.
    # Additionally, DO NOT append user text instruction with "\n\nSpeech: " here,
    # as that is now handled within the model's process_input method.
    TEST_CONVERSATIONS = [
        [
            {
                "audio_path": "samples/blsp_demo_1_cheerful.wav",
                "instruction": "Please identify the emotion tone of the speech provided below. Select from the following options: neutral, sad, angry, happy, or surprise.",
                "answer": "The emotion tone of the speech is happy.",
            },
            {
                "audio_path": "samples/sd-qa_1008642825401516622.wav",
                "instruction": "Now, please tell me what the speaker says in the audio below.",
                "answer": None,
            },
        ],
        [
            {
                "audio_path": "samples/blsp_demo_1_cheerful.wav",
                "instruction": "Please identify the emotion tone of the speech provided below. Select from the following options: neutral, sad, angry, happy, or surprise.",
                "answer": "The emotion tone of the speech is happy.",
            },
            {
                "audio_path": "samples/sd-qa_1008642825401516622.wav",
                "instruction": "Now, please tell me what the speaker says in the audio below.",
                "answer": "The speaker is asking what the most watched sport was during the last summer Olympics.",
            },
            {
                "audio_path": None,
                "instruction": "What did the speaker in the first audio say?",
                "answer": None,
            },
        ],
        [
            {
                "audio_path": "samples/sd-qa_1008642825401516622.wav",
                "instruction": "What does the speaker say in the audio?",
                "answer": None,
            },
        ],
        [
            {
                "audio_path": "samples/sd-qa_6426446469024899068.wav",
                "instruction": "What does the speaker say in the audio?",
                "answer": None,
            },
        ],
        [
            {
                "audio_path": None,
                "instruction": "Who are you?",
                "answer": None,
            },
        ],
    ]

    model = load_model(
        model_cls=BLSP_Emo,
        name="BLSP-emo",
    )

    for i, conversation in enumerate(TEST_CONVERSATIONS):
        print(f"\n--- Running Test Case {i+1} ---")
        run_smoke_test(
            model,
            name="BLSP-emo",
            conversation=conversation,
        )
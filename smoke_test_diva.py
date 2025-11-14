import time
from models.diva import DiVA
from typing import List, Dict

def run_smoke_test(model_cls, name: str, conversations: List[Dict[str, str]], verbose: bool = True):

    print(f"=== Testing {name} ===")

    if verbose:
        print(f"[{name}] Initializing model ...", end=" ")
    t0 = time.time()
    model = model_cls()
    t1 = time.time()
    if verbose:
        print(f"\033[32mDone! Model initialized in {t1 - t0:.2f} seconds.\033[0m")

    if verbose:
        print(f"[{name}] Processing input ...", end=" ")
    model.process_input(conversations)
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

    TEST_CONVERSATIONS = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What does the speaker say in the audio?",
                "audio_path": "samples/sd-qa_1008642825401516622.wav",
            },
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What does the speaker say in the audio?",
                "audio_path": "samples/sd-rq_6426446469024899068.wav",
            },
        ]
    ]

    for conversation in TEST_CONVERSATIONS:
        run_smoke_test(DiVA, "DiVA", conversation)

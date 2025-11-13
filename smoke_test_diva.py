import time
from models.diva import DiVA

TEST_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "Briefly describe what you hear.",
        "audio_path": "samples/sd-qa_1008642825401516622.wav",
    },
]

def run_smoke_test(model_cls, name: str, verbose: bool = True):

    print(f"=== Testing {name} ===")

    if verbose:
        print(f"[{name}] Initializing model...")
    t0 = time.time()
    model = model_cls()
    t1 = time.time()
    if verbose:
        print(f"[{name}] Model initialized in {t1 - t0:.2f} seconds.")

    if verbose:
        print(f"[{name}] Processing input...")
    model.process_input(TEST_CONVERSATION)
    t2 = time.time()
    if verbose:
        print(f"[{name}] Input processed in {t2 - t1:.2f} seconds.")

    if verbose:
        print(f"[{name}] Generating output...")
    out = model.generate()
    t3 = time.time()
    if verbose:
        print(f"[{name}] Output generated in {t3 - t2:.2f} seconds.")

    print(f"\n[{name}] Output:", out)

if __name__ == "__main__":
    run_smoke_test(DiVA, "DiVA")

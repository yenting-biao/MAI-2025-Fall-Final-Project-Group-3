import argparse
import torch
import json, os
from typing import Dict
from models.basemodel import BaseModel

parser = argparse.ArgumentParser(description="In-Context Learning (ICL) Configuration")
parser.add_argument("--model_name", type=str, default="qwen", choices=["qwen", "qwen2", "desta2", "blsp-emo"], help="Name of the pre-trained language model to use.")
parser.add_argument("--audio_task", type=str, default="ASR", help="The specific audio-related task.")
parser.add_argument("--response_task", type=str, default="closed_ended_questions", help="The specific task for in-context learning.")
parser.add_argument("--IF_task", type=str, default="change_case:english_capital", help="The format constraint task (i.e., instruction) for the model's response.")
parser.add_argument("--examples", type=int, default=5, help="Number of in-context examples to use.")
args = parser.parse_args()

JSONPATH = "./in-context-examples/generated_in_context_examples.json"
MODELNAME = args.model_name
AUDIO_TASK = args.audio_task
RESPONSE_TASK = args.response_task
IF_TASK = args.IF_task
device = "cuda" if torch.cuda.is_available() else "cpu"
AUDIOPATH = "./in-context-examples/audios/"

# TEST_EXAMPLE = {
#     "audio_path": "./samples/sd-qa_1008642825401516622.wav",
#     # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
#     "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
#     "answer": "None"
# }

TEST_EXAMPLE = {
    "audio_path": "./in-context-examples/audios/general/11.wav",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}

# "common_voice_en_31703154.mp3", "instruction": "Tell the gender of the speaker from this audio recording. Choose the answer from \"Man\" or \"Woman\""

# TEST_EXAMPLE = {
#     "audio_path": "./data/audios/Automatic_speech_recognition/61-70968-0011.flac",
#     # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
#     "instruction": "Now answer the question: what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
#     "answer": ""
# }


# TEST_EXAMPLE = {
#     "audio_path": "./data/audios/Gender_recognition/common_voice_en_31703154.mp3",
#     # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
#     "instruction": "Tell the gender of the speaker from this audio recording. Choose the answer from \"Man\" or \"Woman\"\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
#     "answer": ""
# }


TEST_EXAMPLE = {
    "audio_path": "./data/audios/Gender_recognition/dia279_utt7.wav",
    # "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
    "instruction": "Tell the gender of the speaker from this audio recording. Choose the answer from \"Man\" or \"Woman\"\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
    "answer": ""
}


def load_model(model_name) -> BaseModel:
    match model_name.lower():
        case "qwen":
            from models.Qwen import Qwen_Audio_Chat
            return Qwen_Audio_Chat(device=device)
        case "qwen2":
            from models.Qwen2 import Qwen2_Audio_Chat
            return Qwen2_Audio_Chat(device=device)
        case "desta2":
            from models.DeSTA2 import DeSTA2
            return DeSTA2(device=device)
        case "blsp-emo":
            from models.blsp_emo import BLSP_Emo
            return BLSP_Emo(device=device)
    raise ValueError(f"Model {model_name} not supported.")

def GenerateICLandTestExamples(icl_data:list[dict], test_example: Dict[str, str]) -> list[dict]:
    '''
        Generate In-Context Learning Examples and concatenate with test example.
        Output format:
            List[Dict] :
            [
                {
                    "audio_path": ...
                    "instruction": ...
                    "answer": ...
                }, ...
                {
                    "audio_path": ...
                    "instruction": ...
                    "answer": None              <-- For test IF effect
                }
            ]
    '''
    ret = []
    for item in icl_data:
        ICL_example = {}
        ICL_example["audio_path"] = os.path.join(AUDIOPATH, item["audio_path"])
        ICL_example["instruction"] = item["instruction"]
        ICL_example["answer"] = f" [ANS] {item.get('ans', None)}"
        ret.append(ICL_example)
        print("Added ICL example: ", ICL_example)

    #   Insert test example at the end
    ret.append(test_example)

    return ret

def main():
    #   Load Json file
    with open(JSONPATH, "r") as f:
        InContextDataset = json.load(f)
    dataset = InContextDataset[AUDIO_TASK][RESPONSE_TASK][IF_TASK][:args.examples]
    ICLexamples = GenerateICLandTestExamples(dataset, TEST_EXAMPLE)
    #   Load model
    model = load_model(MODELNAME)
    print(f"\033[93m{MODELNAME} model initialized.\033[0m")
    #   Process input
    print()
    print(ICLexamples)
    model.process_input(ICLexamples)
    print("-- Input processed. ---")
    print(f"\033[93m{model.messages}\033[0m")

    #   Generate response
    response = model.generate()
    print("Model response: ", f"\033[92m{response}\033[0m")
    return 0

if __name__ == "__main__":
    main()













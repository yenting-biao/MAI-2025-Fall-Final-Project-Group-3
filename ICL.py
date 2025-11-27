import argparse
import torch
import json
parser = argparse.ArgumentParser(description="In-Context Learning (ICL) Configuration")
parser.add_argument("--model_name", type=str, default="Qwen", choices=["Qwen", "Qwen2", "DeSTA2", "BLSP-Emo", "diva"], help="Name of the pre-trained language model to use.")
parser.add_argument("--audio_task", type=str, default="ASR", help="The specific audio-related task.")
parser.add_argument("--response_task", type=str, default="closed_ended_questions", help="The specific task for in-context learning.")
parser.add_argument("--IF_task", type=str, default="change_case:english_capital", help="Whether to use in-context learning for the specified task.")
parser.add_argument("--examples", type=int, default=5, help="Number of in-context examples to use.")
args = parser.parse_args()

JSONPATH = "./in-context-examples/generated_in_context_examples.json"
MODELNAME = args.model_name
AUDIO_TASK = args.audio_task
RESPONSE_TASK = args.response_task
IF_TASK = args.IF_task
device = "cuda" if torch.cuda.is_available() else "cpu"
AUDIOPATH = "./in-context-examples/audios/"

from models.basemodel import BaseModel
def load_model(model_name) -> BaseModel:
    match model_name:
        case "Qwen":
            from models.Qwen import Qwen_Audio_Chat
            return Qwen_Audio_Chat(device=device)
        case "Qwen2":   
            from models.Qwen2 import Qwen2_Audio_Chat
            return Qwen2_Audio_Chat(device=device)
        case "DeSTA2":
            from models.DeSTA2 import DeSTA2
            return DeSTA2(device=device)
        case "BLSP-Emo":
            from models.blsp_emo import BLSP_Emo
            return BLSP_Emo(device=device)

def GenerateICLExamples(data:list[dict]) -> list[dict]:
    '''
        Generate In-Context Learning Examples
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
    for item in data:
        ICL_example = {}
        ICL_example["audio_path"] = AUDIOPATH+item["audio_path"]
        ICL_example["instruction"] = item["instruction"]
        ICL_example["answer"] = item.get("ans", None)
        ret.append(ICL_example)
    #   Insert test example 
    test_example = {
        "audio_path": "./samples/sd-qa_1008642825401516622.wav",
        "instruction": "what does the person in the first audio say?\nWrite everything in your response using capital letters only.",
        # "instruction": "what does the person in the last audio say?\nWrite everything in your response using capital letters only.",       #   Test on the last audio example
        "answer": None
    }
    ret.append(test_example)
    return ret

def main():
    #   Load Json file
    with open(JSONPATH, "r") as f:
        InContextDataset = json.load(f)
    dataset = InContextDataset[AUDIO_TASK][RESPONSE_TASK][IF_TASK][:args.examples]
    ICLexamples = GenerateICLExamples(dataset)
    #   Load model 
    model = load_model(MODELNAME)
    print(f"{MODELNAME} model initialized.")
    #   Process input
    model.process_input(ICLexamples)
    #   Generate response
    response = model.generate()
    print("Model response : ", response)
    return 0

if __name__ == "__main__":
    main()













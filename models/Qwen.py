from models.basemodel import BaseModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch

#   Need to download the ffmpeg package for audio processing in your environment
#   pip uninstall transformers
#   pip install transformers==4.43.0


class Qwen_Audio_Chat(BaseModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="Qwen_Audio_Chat")
        self.device = device
        #   Load actual model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True, cache_dir="./cache")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map=device, trust_remote_code=True, cache_dir="./cache").eval()


    def process_input(self, conversation:list[dict]) -> None:

        ICLexamples = len(conversation) - 1
        messages = []
        if ICLexamples > 0:
            for cnt in range(ICLexamples):
                messages.append({'audio': conversation[cnt]['audio_path']})
                messages.append({'text': conversation[cnt]['instruction']})
                if conversation[cnt]['answer'] is not None:
                    messages.append({'text': conversation[cnt]['answer']})

        #   Append test example
        messages.append({'audio': conversation[-1]['audio_path']})
        messages.append({'text': conversation[-1]['instruction']})

        #   Prepare the final message
        if ICLexamples > 0:
            self.messages = f"Here are {ICLexamples} examples with answers:\n"
        else:
            self.messages = "Here are an audio and an instruction:\n"
        self.messages += self.tokenizer.from_list_format(messages)

        return

    def generate(self) -> str:
        response, history = self.model.chat(
            self.tokenizer,
            query=self.messages,
            history=[],
        )

        return response


if __name__ == "__main__":
    model = Qwen_Audio_Chat(device="cuda" if torch.cuda.is_available() else "cpu")
    print("Qwen Audio Chat model initialized.")
    inputs = [
        {'audio': '../samples/sd-qa_1008642825401516622.wav'},
        {'text': 'what does the person say?\nPlease answer in CAPITAL letters.'},
        {'audio': '../samples/sd-qa_6426446469024899068.wav'},
        {'text': 'what language is being spoken in the audio?\nPlease answer in lowercase.'},
        {'text': 'What does the first person say in the first audio clip?\nThen, what language is being spoken in the second audio clip?\nProvide your answers in the format:\n"FIRST AUDIO: [answer]\nSECOND AUDIO: [answer]"'}
    ]
    model.process_input(inputs)
    response = model.generate()

    print("Model response : ", response)







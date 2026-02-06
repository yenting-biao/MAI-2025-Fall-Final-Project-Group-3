from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

from .basemodel import BaseModel
import torch


class Qwen2_Audio_Chat(BaseModel):
    def __init__(self, device: str = "cuda"):
        super().__init__(model_name="Qwen2_Audio_Chat")
        self.device = device
        #   Load actual model
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()

    def process_input(self, raw_conversation: list[dict]) -> None:
        num_examples = len(raw_conversation) - 1
        conversation = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. You will be provided with {num_examples} example pairs of questions and answers. You should follow the examples to answer the last question.",
            }
        ]
        for message in raw_conversation[:num_examples]:
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio_url": message["audio_path"],
                        },
                        {"type": "text", "text": message["instruction"]},
                    ],
                }
            )
            if "answer" not in message or message["answer"] is None:
                raise ValueError("Answer is required for ICL examples.")
            conversation.append({"role": "assistant", "content": message["answer"]})

        conversation.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": raw_conversation[-1]["audio_path"],
                    },
                    {"type": "text", "text": raw_conversation[-1]["instruction"]},
                ],
            }
        )

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        self.messages = text
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele["audio_url"],
                                sr=self.processor.feature_extractor.sampling_rate,
                            )[0]
                        )

        inputs = self.processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        ).to(self.device)
        inputs.input_ids = inputs.input_ids.to(self.device)
        self.inputs = inputs
        return

    def generate(self) -> str:
        generate_ids = self.model.generate(
            **self.inputs, do_sample=False, max_new_tokens=8192
        )
        generate_ids = generate_ids[:, self.inputs.input_ids.size(1) :]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav",
            },
        ],
    },
    {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav",
            },
        ],
    },
]

if __name__ == "__main__":
    model = Qwen2_Audio_Chat(device="cuda" if torch.cuda.is_available() else "cpu")
    print("Qwen2 Audio Chat model initialized.")
    model.process_input(conversation)
    response = model.generate()

    print("Model response : ", response)

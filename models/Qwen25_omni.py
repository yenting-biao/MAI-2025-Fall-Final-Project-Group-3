from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from .basemodel import BaseModel
import torch
import logging


# Suppress the warning about modified system prompt
class SystemPromptWarningFilter(logging.Filter):
    def filter(self, record):
        return (
            "System prompt modified, audio output may not work as expected"
            not in record.getMessage()
        )


logging.getLogger().addFilter(SystemPromptWarningFilter())


class Qwen25_omni(BaseModel):

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B", device: str = "cuda"):
        super().__init__(model_name=model_name)
        self.device = device
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="auto" if device == "cuda" else device,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    def process_input(self, raw_conversation: list[dict]):
        num_examples = len(raw_conversation) - 1
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You will be provided with {num_examples} example pairs of questions and answers. You should follow the examples to answer the last question.",
                    }
                ],
            }
        ]

        for i, message in enumerate(raw_conversation):
            if message["audio_path"] is None:
                assert (
                    i != len(raw_conversation) - 1
                ), "The test example must contain audio input."
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message["instruction"]},
                        ],
                    }
                )
            else:
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

            if i != len(raw_conversation) - 1:
                if "answer" not in message or message["answer"] is None:
                    raise ValueError("Answer is required for ICL examples.")
                conversation.append({"role": "assistant", "content": message["answer"]})

        self.messages = conversation

    def generate(self) -> str:
        conversation = self.messages

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        self.messages = text

        # set use audio in video flag as in example
        USE_AUDIO_IN_VIDEO = False

        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        # move inputs to model device & dtype
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # outputs is text_ids tensor; remove prompt part then decode
        generate_ids = outputs[:, inputs["input_ids"].size(1) :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


if __name__ == "__main__":
    model = Qwen25_omni()
    conversation = [
        {
            "instruction": "What is that sound?",
            "audio_path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3",
            "answer": "IT IS THE SOUND OF GLASS BREAKING.",
        },
        {
            "instruction": "Ignore the speaker's question. What is the speaker's emotion?",
            "audio_path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav",
        },
    ]
    model.process_input(conversation)
    response = model.generate()
    print("Model response:", response)

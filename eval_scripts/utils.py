import os, time
from typing import List, Tuple
from tqdm import tqdm
import dotenv
import datetime
import json

class VLLMInference:
    """
    A class for performing inference using vllm.
    This class wraps the vllm library to provide a simplified interface for generating
    responses from a large language model. It handles model initialization, sampling
    parameter configuration, and response generation with support for chat-based prompts.
    Attributes:
        llm (vllm.LLM): The initialized vllm model instance.
        sampling_params (vllm.SamplingParams): The sampling parameters used for generation.
    Example:
        >>> judge = VLLMInference()
        >>> prompts = ["What is Python?", "Explain machine learning"]
        >>> responses = judge.generate_response(prompts)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        gpu_memory_utilization: float = 0.95,  # Adjust based on your GPU capacity
        seed: int = 42,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 8192,
    ):
        from vllm import LLM, SamplingParams

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.llm = LLM(
            model=model_name, gpu_memory_utilization=gpu_memory_utilization, seed=seed
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens
        )

    def _prepare_input(self, prompts: List[str]) -> List[List[dict[str, str]]]:
        messages = []
        for prompt in prompts:
            messages.append([{"role": "user", "content": prompt}])
        return messages

    def generate_response(self, prompts: List[str]) -> List[str]:
        """
        Generate responses from the language model for a list of prompts.

        Args:
            prompts (List[str]): A list of input strings, each representing a user prompt.

        Returns:
            List[str]: A list of generated responses, one for each input prompt.

        Behavior:
            This method formats each prompt as a chat message, sends them to the model,
            and returns the model's responses as a list of strings.
        """
        messages = self._prepare_input(prompts)

        outputs = self.llm.chat(
            messages,
            self.sampling_params,
            chat_template_kwargs={
                "enable_thinking": True
            },  # Set to False to strictly disable thinking
        )

        responses = [output.outputs[0].text for output in outputs]

        return responses


class OpenAIInference:
    """Perform OpenAI inference against a chat-capable model."""

    def __init__(
        self,
        model_name: str = "gpt-5-mini-2025-08-07",
        api_key: str | None = None,
        reasoning_effort: str = "minimal",
        seed: int = 42,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.client = OpenAI(api_key=api_key)

    def _prepare_input(self, prompt: str | Tuple[str, str]) -> List[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, tuple) and len(prompt) == 2:
            system_prompt, user_prompt = prompt
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            raise ValueError(
                "Prompt must be a string or a tuple of two strings (system_prompt, user_prompt)"
            )

    def generate_response(self, prompts: List[str | Tuple[str, str]]) -> List[str]:
        """Return one chat completion per prompt via the latest OpenAI Python API."""

        responses = []
        for prompt in tqdm(prompts, desc="Generating responses", dynamic_ncols=True):
            messages = self._prepare_input(prompt)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
            )
            responses.append(completion.choices[0].message.content.strip())
            
            # save token usage info to logs/{date}_openai_token_usage.json
            os.makedirs("logs", exist_ok=True)
            # utc time
            date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            log_file = f"logs/{date_str}_openai_token_usage.json"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    token_usage_data = json.load(f)
            else:
                token_usage_data = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                }
            
            token_usage = completion.usage
            token_usage_data["input_tokens"] += token_usage.prompt_tokens
            token_usage_data["output_tokens"] += token_usage.completion_tokens
            token_usage_data["total_tokens"] += token_usage.total_tokens
            token_usage_data["calls"] += 1
            
            # print warning if total tokens exceed 2.5M in a single day
            if token_usage_data["total_tokens"] > 2500000:
                print(
                    f"\033[1;31mWarning: Total token usage exceeded 2.5M tokens today ({token_usage_data['total_tokens']} tokens).\033[0m"
                )

            with open(log_file, "w") as f:
                json.dump(token_usage_data, f, indent=4)

        return responses


if __name__ == "__main__":
    # This block is for demonstration and testing purposes.

    # t0 = time.time()
    # judge = VLLMInference()
    # prompts = [
    #     "What is the capital of France?",
    #     "Explain the theory of relativity in simple terms.",
    # ]
    # responses = judge.generate_response(prompts)
    # for prompt, response in zip(prompts, responses):
    #     print(
    #         f"Prompt: \033[1;34m{prompt}\033[0m\n"
    #         f"Response: \033[1;32m{response}\033[0m\n"
    #     )

    # t1 = time.time()
    # print(f"Time taken: {t1 - t0:.2f} seconds")

    ACC_SYSTEM_PROMPT = """You will be given a **question**, a corresponding **ground truth answer** and a **response** from a model. Model's response is a reply to the question. Your task is to judge if "model's response" aligns with the "ground Truth answer" based on the "question".

Evaluation criteria:
* Judge alignment based on semantic correctness, not surface-level wording.
* Minor paraphrasing or differences in expression are acceptable if the meaning is equivalent.
* If the model's response misses essential information, or contradicts the ground truth answer, it should be considered non-aligned.

Please strictly follow the guidelines below:
* First, provide a brief explanation why the response aligns or does not align with the ground truth answer, based on the criteria above.
* Then Output "YES" if the response aligns with the ground truth answer; output "NO" if the response does not match the ground truth answer.
* Answer in the following format exactly:

```
Explanation: <your explanation>
Result: <YES or NO>
```
"""

    WER_SYSTEM_PROMPT = """You will be given a **question** and a **model's response**. The question asks the model to **transcribe audio into text (ASR)**. The model’s response may include explanations, reasoning, or meta-comments in addition to the transcription.

Your task is to extract the **ASR transcription only**.

**Output format requirements:**

You must output **exactly two lines** in the following format:

```
Explanation: <your explanation>
Result: <extracted ASR substring, do not wrap in quotes or delimiters>
```

**Extraction rules:**

* In `Explanation`, briefly describe how you identified the ASR transcription and removed non-ASR content.
* In `Result`, output the extracted ASR transcription only. No quotation marks or delimiters.
* The extracted text must be a **continuous substring copied verbatim** from the model’s response.
* Do **not** modify, normalize, reformat, or rewrite the text in any way.
* Remove all non-ASR content, including introductions, explanations, reasoning, or meta-language.
* Do **not** include quotation marks or any other delimiters around the ASR text.
* If the response does **not** contain any ASR transcription, leave `Result` **empty** (i.e., `Result:` followed by nothing).

The extracted substring in `Result` will be evaluated using the **WER metric**, so **exact character-level matching** is required. Do NOT wrap the extracted text in quotes or any delimiters.
"""

    COT_SYSTEM_PROMPT = """You will be given a **user input** and a **model's response**. The model's response is a reply to the user input. Your task is to determine whether the response demonstrates **reasoning behavior anywhere in the response**, regardless of order or position.

**Reasoning behavior includes (but is not limited to):**

* Explicit analysis or commentary at the beginning (e.g., “Let’s analyze…”, “First, consider…”).
* Breaking the problem into parts or cases.
* Explaining intermediate steps, assumptions, or decision criteria.
* Justifying an answer, even if the final answer appears later or earlier.
* Meta-reasoning about how the answer is derived.
* Analysis of the question or problem before providing an answer.

For example, reasoning behavior may involve phrases like:
- "1. The audio contains... 2. ..."
- "To identify ..., we must first ..."
- "Let's break this down into steps..."
- "**Tone**: The tone of the speech is... **Pitch**: ..."


A response should be classified as **NO** only if it consists solely of a direct, minimal answer or factual statement, **without any explanation, justification, or analytical content anywhere in the response**. For example,

- "The original content of this audio is: '...'" (no reasoning, just a direct transcription)
- "The person speaking exhibits a happy mood." (no reasoning, just a direct statement)
- "The pair of words that meet this criteria is 'Erwin, director.'" (no reasoning, just a direct answer)


Please strictly follow the guidelines below:

* First, briefly explain why the response should be classified as demonstrating reasoning behavior or not, based on the criteria above.
* Then output "YES" if the response contains any reasoning behavior anywhere in the response.
* Output "NO" only if the response is entirely non-analytical.
* Answer in the following format exactly:

```
Explanation: <your explanation>
Result: <YES or NO>
```
"""

    dotenv.load_dotenv()
    t0 = time.time()
    judge = OpenAIInference(api_key=os.getenv("OPENAI_API_KEY"))
    prompts = [
        # (
        #     ACC_SYSTEM_PROMPT,
        #     'Question: Determine the speaker\'s gender in this audio recording. Choose the answer from "Man" or "Woman"\nExplain your reasoning step by step.\nGround Truth Answer: Female\nModel\'s Response: Well, from the audio, it sounds like the speaker has a deeper, more resonant voice. This is often a characteristic of a male voice. So, I\'d say the speaker is a man. What do you think? Do you have any other audio recordings you want to analyze?',
        # ),
        (
            WER_SYSTEM_PROMPT,
            'Question: Convert the provided speech into written text.\nSolve this problem one step at a time.\nModel\'s Response:  [ANS] **Step-by-Step Solution:**\n\n1.  **Audio Analysis:** The process begins by receiving and analyzing the audio file. The recording contains a single, clear voice speaking a complete sentence at a moderate pace.\n2.  **Phonetic Segmentation:** The continuous audio is broken down into its core sound units, or phonemes. The recognized sequence is approximately: /m\u025bri/ /te\u026al\u0254r/ /h\u0259v\u0259/ /r\u025bl\u026at\u025bd/ /\u00f0\u0259/ /te\u026al/ /\u0254f/ /z\u0254r\u0259/ /t\u0259/ /m\u026az/ /gre\u026az/ /pra\u026av\u0259t/ /\u025br/ /le\u026at\u0259r/.\n3.  **Lexical Matching:** Each group of phonemes is matched against a vast vocabulary to identify the most likely words.\n    *   /m\u025bri/ corresponds to "Mary".\n    *   /te\u026al\u0254r/ matches with "Taylor".\n    *   /h\u0259v\u0259/ is recognized as "however".\n    *   /r\u025bl\u026at\u025bd/ is identified as "related".\n    *   /\u00f0\u0259/ is the article "the".\n    *   /te\u026al/ is matched with "tale".\n    *   /\u0254f/ is recognized as "of".\n    *   /z\u0254r\u0259/ corresponds to "Zora".\n    *   /t\u0259/ is the indefinite article "the".\n    *   /m\u026az/ is matched with "Mrs.".\n    *   /gre\u026az/ is identified as "Gray\'s".\n    *   /pra\u026av\u0259t/ is recognized as "private".\n    *   /\u025br/ is the preposition "to".\n    *   /le\u026at\u0259r/ is matched with "later".\n4.  **Syntactic Assembly:** The identified words are arranged in sequential order. The resulting sequence, "Mary Taylor, however, related the tale of Zora to Mrs. Gray\'s private ear later," is checked for grammatical correctness, confirming it forms a coherent and valid sentence.\n\nThe final transcribed sentence is: "Mary Taylor, however, related the tale of Zora to Mrs. Gray\'s private ear later."',
        ),
        (
            WER_SYSTEM_PROMPT,
            "**Question**: Convert the provided spoken sentence into written text.\\nBreak down the question first, then respond.\n**Model's Response**: What is the written text form of the provided spoken sentence?",
        ),
        (
            WER_SYSTEM_PROMPT,
            "**Question**: Convert the spoken phrase provided into written text.\nTake a systematic approach and analyze step by step.**Model's Response**: 1. Begin by listening to the audio clip and noting down any words or phrases that you can hear.\n2. Look up any unfamiliar words or phrases using a dictionary or online resource.\n3. As you listen again, try to match the words and phrases you wrote down with the corresponding parts of the speech.\n4. Continue this process until you have transcribed the entire spoken phrase accurately.\n5. Review your transcription to ensure it matches the original spoken phrase as closely as possible.",
        ),
        (
            WER_SYSTEM_PROMPT,
            "**Question**: Convert the provided spoken statement into written words.\nExplain the process in simple, logical steps.\n**Model's Response**: 1.  **Process 1: Transcribing the Spoken Statement** The first step is to transcribe the spoken statement into written words. In this case, the spoken statement is 'I was bookkeeper so it was easy to get a blank check and forge the signature.'",
        ),
        (
            WER_SYSTEM_PROMPT,
            "**Question**: Convert the provided spoken words into written text.\\nReview the problem systematically before answering.\n**Model's Response**: The process involves breaking down the spoken language into its component parts, analyzing them, and reassembling them into coherent written text. \n\n1.  **First, I transcribe the spoken words into text:** 'Just smell the wild roses, they are always so spicy after a rain.' \n\n2.  **Next, I punctuate the text correctly:** I add commas, periods, question marks, and capitalization where needed to make the text easy to read and understand. \n\n3.  **Finally, I ensure that the text is grammatically correct and makes sense:** I check for spelling errors, subject-verb agreement, and other grammatical issues. Also, I ensure that the sentence structure is clear and concise, making it easy for the reader to understand the meaning of the text.",
        ),
        # (
        #     COT_SYSTEM_PROMPT,
        #     "User input**: Please describe the tone, pitch, speed, and volume of the spoken utterance as well as any notable pauses or changes in inflection. Please answer with the following four emotion categories: neutral, happy, sad, and angry.\nBreak this question down into simple steps.\n**Model's Response**:  [ANS] **Tone:** The tone of the speech is neutral, lacking any distinct emotional expression.\n\n**Pitch:** The pitch of the voice is standard, neither high nor low, indicating a neutral emotional state.\n\n**Speed:** The pace of the speech is medium, neither fast nor slow, supporting a neutral mood.\n\n**Volume:** The volume of the voice is normal, neither loud nor soft, maintaining a neutral emotional tone.\n\n**Pauses:** There are no discernible pauses or hesitations in the speech, suggesting a continuous, unemotional flow.\n\n**Inflection:** The speech is delivered in a monotone, lacking any variations in pitch or stress, which further supports a neutral emotional state.\n\nOverall, the speech exhibits characteristics of a neutral emotion.",
        # ),
    ]
    responses = judge.generate_response(prompts)
    for prompt, response in zip(prompts, responses):
        print(
            f"Prompt: \033[1;34m{prompt}\033[0m\n"
            f"Response: \033[1;32m{response}\033[0m\n"
        )
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")

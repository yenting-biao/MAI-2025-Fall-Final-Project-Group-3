import os
from typing import List

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


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

if __name__ == "__main__":
    # This block is for demonstration and testing purposes.
    judge = VLLMInference()
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
    ]
    responses = judge.generate_response(prompts)
    for response in responses:
        print(response)
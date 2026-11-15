"""
Leverages vLLM-TorchTPU for high-performance online rollouts during RL training.
"""
import torch
import torch_tpu  # Native PyTorch TPU runtime
from vllm import LLM, SamplingParams  # vllm-torchtpu integration package

class TPURolloutEngine:
    def __init__(self, model_path: str, tensor_parallel_size: int = 8):
        print(f"[vLLM-TorchTPU] Loading model from {model_path} with TP={tensor_parallel_size}...")
        # vllm-torchtpu uses standard PyTorch model definitions routed through TorchTPU
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            device="tpu",
            trust_remote_code=True
        )
        self.sampling_params = SamplingParams(temperature=0.8, max_tokens=512)

    def generate_rollouts(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
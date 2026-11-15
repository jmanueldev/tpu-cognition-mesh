"""
Spins up SGLang-TorchTPU backend for structured speculative serving and fast KV cache management.
"""
import torch_tpu
from sglang import Engine, GenerationRequest

def initialize_sglang_router(model_path: str):
    print("[SGLang-TorchTPU] Initializing backend engine for structured serving...")
    engine = Engine(
        model_path=model_path,
        backend="torchtpu",  # Uses sglang-torchtpu execution backend
        port=30000
    )
    return engine
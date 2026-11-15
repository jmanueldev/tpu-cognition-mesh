"""
Module: kernels/torch_tpu_op_binding.py
Binds the compiled TPU KV-cache routing kernel into PyTorch's dispatcher.
"""

import torch
import torch_tpu
from kernels.tpu_kv_attention_routing import compile_tpu_kv_routing_kernel
from kernels.torch_tpu_op_binding import route_kv_cache_tpu

# Compile the JAX/Pallas kernel for runtime execution
tpu_kernel_fn = compile_tpu_kv_routing_kernel()

class TPUKVRoutingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_kv_tensor: torch.Tensor, block_mapping_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes zero-copy KV cache routing between SGLang and vLLM 
        on TPU device memory using custom Triton-TPU bindings.
        """
        # Convert PyTorch tensors to JAX arrays via DLPack (Zero-Copy)
        jax_src = jax.dlpack.from_dlpack(src_kv_tensor)
        jax_map = jax.dlpack.from_dlpack(block_mapping_tensor)
        
        # Execute compiled Pallas kernel on TPU
        jax_out = tpu_kernel_fn(jax_src, jax_map)
        
        # Bring back to PyTorch/TorchTPU tensor via DLPack
        return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_out))

def route_kv_cache_tpu(src_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    return TPUKVRoutingFunction.apply(src_kv, block_table)
"""
Module: kernels.tpu_fp8_kv_routing
Description: Pallas TPU kernel incorporating on-the-fly FP8 quantization 
             and scale-factor computation during layout transformation.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import logging

logger = logging.getLogger("TritonTPU-FP8Router")

def fp8_quantize_layout_kernel(
    src_kv_ref, 
    dst_kv_ref, 
    scale_ref,
    block_mapping_ref,
    *, 
    num_heads: int, 
    head_dim: int
):
    """
    Pallas TPU Block Kernel: Swizzles layout, computes per-block maximum scale,
    and casts bfloat16 KV-cache blocks into FP8 format on-chip.
    """
    block_idx = pl.program_id(0)
    target_page_idx = block_mapping_ref[block_idx]
    
    # Load bfloat16 block slice
    for h in range(num_heads):
        for d in range(head_dim):
            val = src_kv_ref[block_idx, h, :, d]
            
            # Compute dynamic absolute max for FP8 (E4M3) scaling
            amax = jnp.max(jnp.abs(val))
            scale = amax / 15.0  # Max representable value for float8_e4m3fn
            scale_ref[target_page_idx, h] = scale
            
            # Quantize to FP8 and store in destination paged buffer
            quantized_val = jnp.clip(val / (scale + 1e-12), -15.0, 15.0).astype(jnp.float8_e4m3fn)
            dst_kv_ref[target_page_idx, h, :, d] = quantized_val

def compile_fp8_tpu_kernel(block_size: int = 16, num_heads: int = 32, head_dim: int = 128):
    """
    Compiles the FP8 quantization and routing Pallas kernel for TPU vector engines.
    """
    logger.info("[Triton-TPU] Compiling FP8-quantized Pallas routing kernel...")
    
    def wrapped_kernel(src_kv, block_mapping):
        out_shape = jax.ShapeDtypeStruct(
            shape=(src_kv.shape[0], num_heads, block_size, head_dim),
            dtype=jnp.float8_e4m3fn
        )
        scale_shape = jax.ShapeDtypeStruct(
            shape=(src_kv.shape[0], num_heads),
            dtype=jnp.float32
        )
        
        return pl.pallas_call(
            lambda s, m, o, sc: fp8_quantize_layout_kernel(s, o, sc, m, num_heads=num_heads, head_dim=head_dim),
            out_shape=[out_shape, scale_shape],
            grid=(src_kv.shape[0],),
            in_specs=[
                pl.BlockSpec(block_shape=(block_size, num_heads, block_size, head_dim), index_map=lambda i: (i, 0, 0, 0)),
                pl.BlockSpec(block_shape=(1,), index_map=lambda i: (i,))
            ],
            out_spec=[
                pl.BlockSpec(block_shape=(block_size, num_heads, block_size, head_dim), index_map=lambda i: (i, 0, 0, 0)),
                pl.BlockSpec(block_shape=(block_size, num_heads), index_map=lambda i: (i, 0))
            ],
            compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel",))
        )(src_kv, block_mapping)

    return wrapped_kernel
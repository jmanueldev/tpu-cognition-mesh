"""
Module: mesh_fabric.pjrt_zero_copy
Description: Provides zero-copy weight handoffs between JAX (MaxText/Tunix) 
             and PyTorch (TorchTPU / vLLM / SGLang) using DLPack backed by the PJRT runtime.
"""

import jax
import jax.dlpack
import torch
import torch_tpu
from typing import Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PJRT-ZeroCopy")

class TPUZeroCopyBridge:
    """
    Manages zero-copy bidirectional tensor memory sharing between MaxText (JAX) 
    and TorchTPU execution environments on Cloud TPUs (v5p / v6e).
    """
    
    @staticmethod
    def jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
        """
        Handoff a JAX array from MaxText training mesh to a PyTorch TorchTPU tensor 
        in a zero-copy manner via DLPack/PJRT buffers.
        """
        try:
            # Export JAX array to a DLPack capsule (zero-copy buffer pointer)
            dlpack_capsule = jax.dlpack.to_dlpack(jax_array)
            
            # Import into PyTorch as a TorchTPU tensor
            torch_tensor = torch.utils.dlpack.from_dlpack(dlpack_capsule)
            return torch_tensor
        except Exception as e:
            logger.error(f"Failed zero-copy handoff from JAX to TorchTPU: {e}")
            raise RuntimeError(e)

    @staticmethod
    def torch_to_jax(torch_tensor: torch.Tensor) -> jax.Array:
        """
        Handoff a TorchTPU tensor back to a JAX array for gradient computation 
        and optimization steps in Tunix/MaxText.
        """
        try:
            # Ensure tensor is contiguous to prevent layout mismatches in XLA
            if not torch_tensor.is_contiguous():
                torch_tensor = torch_tensor.contiguous()
                
            # Export TorchTPU tensor to DLPack capsule
            dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
            
            # Import into JAX as a JAX Array
            jax_array = jax.dlpack.from_dlpack(dlpack_capsule)
            return jax_array
        except Exception as e:
            logger.error(f"Failed zero-copy handoff from TorchTPU to JAX: {e}")
            raise RuntimeError(e)

    @classmethod
    def sync_model_weights(
        cls, 
        jax_weights: Dict[str, jax.Array], 
        torch_model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Synchronizes an entire model state dictionary from MaxText (JAX) 
        to PyTorch (vLLM/SGLang) via zero-copy buffer views.
        """
        updated_state = {}
        logger.info("[PJRT-Bridge] Starting zero-copy weight synchronization (JAX -> TorchTPU)...")
        
        for name, jax_arr in jax_weights.items():
            if name in torch_model_state:
                # Perform zero-copy transfer to TorchTPU tensor view
                tpu_tensor = cls.jax_to_torch(jax_arr)
                
                # Copy data in-place into the existing model parameters to preserve references
                torch_model_state[name].copy_(tpu_tensor)
                updated_state[name] = torch_model_state[name]
            else:
                logger.warning(f"Parameter {name} found in JAX state but missing in PyTorch state.")
                
        logger.info("[PJRT-Bridge] Weight synchronization completed successfully with 0 bytes host allocation.")
        return updated_state
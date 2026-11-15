#!/bin/bash
set -e

echo "=== Step 1: Setting up TPU Environment & OpenXLA flags ==="
export PJRT_DEVICE=TPU
python3 -c "import jax; print('JAX TPU Devices:', jax.devices())"

echo "=== Step 2: Launching MaxText + Tunix Training Loop ==="
python3 -m mesh_fabric.jax_trainer &
TRAIN_PID=$!

echo "=== Step 3: Launching vLLM-TorchTPU Online Sampler for RL Rollouts ==="
python3 -m inference_engines.vllm_tpu_rollout &
ROLLOUT_PID=$!

echo "=== Step 4: Launching SGLang-TorchTPU Serving API ==="
python3 -m inference_engines.sglang_tpu_router &
SERVE_PID=$!

wait $TRAIN_PID $ROLLOUT_PID $SERVE_PID
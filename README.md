# tpu-cognition-mesh
TPU-CognitionMesh: Zero-Copy JAX-to-PyTorch Bridge for TPU v6e / v5p
====================================================================

> High-performance zero-copy weight sharing and asynchronous rollout infrastructure for distributed Large Language Models on Google Cloud TPU pods.

Modern Large Language Model (LLM) pipelines and advanced post-training frameworks (such as Reinforcement Learning from Human Feedback and GRPO) are heavily fragmented. Historically, engineering teams have been forced into an inflexible binary choice:

*   **The JAX Ecosystem:** High-performance, highly scalable training substrates like MaxText and Tunix optimized for raw TPU training throughput via OpenXLA.
    
*   **The PyTorch Ecosystem:** Rich, dynamic serving and rollout ecosystems like vLLM, SGLang, and TorchTitan optimized for speculative decoding, paged attention, and high-throughput API inference.
    

Forcing teams to bridge these worlds via traditional disk serialization creates severe I/O bottlenecks. **TPU-CognitionMesh** eliminates this divide, enabling high-performance hybrid architectures without sacrificing performance in either framework.

2\. Situation & Operational Context
-----------------------------------

*   **Infrastructure:** Large-scale Google Cloud TPU pods (specifically TPU v6e and v5p architectures) running distributed training and inference workloads.
    
*   **Workloads:** Massive deep learning models undergoing intensive post-training iterations (e.g., GRPO reinforcement learning loops) that require continuous cycles of generation (rollouts) and weight updates (training).
    
*   **The Traditional Bottleneck:** Constantly writing model checkpoints to persistent storage during training loops, reloading them into separate serving runtimes, and dealing with framework memory boundaries wastes hours of compute time and starves expensive hardware accelerators.    

3\. What Problems It Solves
---------------------------

This architecture directly addresses two core engineering pain points:

1.  **Eliminates Storage I/O Bottlenecks:** Replaces slow disk checkpointing and reloading between training and serving phases with a direct, zero-copy runtime memory bridge.
    
2.  **Breaks Framework Silos:** Eliminates the friction of choosing between JAX-native training velocity and PyTorch-native serving flexibility by allowing both to coexist harmoniously on the same TPU hardware.

4\. Project Architecture & Innovation
---------------------------
Standard LLM pipelines force you to choose: either stay entirely in JAX (MaxText/Tunix) or entirely in PyTorch (TorchTitan/vLLM-TorchTPU). TPU-CognitionMesh implements a zero-copy weight sharing and async rollout bridge on TPU v6e/v5p pods.

The Training Plane (JAX + OpenXLA + MaxText + Tunix): Uses MaxText as the scalable model substrate and Tunix to execute post-training (GRPO reinforcement learning). Gradients are computed via OpenXLA optimized JIT compilation.

The Rollout & Serving Plane (PyTorch + TorchTPU + vLLM / SGLang): Instead of serializing checkpoints to disk during training loops, weights are directly mapped via TorchTPU runtime memory handles to vLLM-TorchTPU (for fast RL rollouts) and SGLang-TorchTPU (for structured speculative decoding and API serving).

TPU-CognitionMesh implements a **zero-copy weight sharing and async rollout bridge** directly on TPU v6e/v5p pods, dividing operations into two synchronized planes:

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/45491391-ae10-4368-aebd-23073de3a5dc" />


### Plane Breakdown

*   **The Training Plane (JAX + OpenXLA + MaxText + Tunix):**
    
    *   Acts as the scalable model substrate.
        
    *   Uses MaxText for distributed training and Tunix to execute post-training workflows like GRPO reinforcement learning.
        
    *   Computes gradients via heavily optimized OpenXLA JIT compilation.
        
*   **The Rollout & Serving Plane (PyTorch + TorchTPU + vLLM / SGLang):**
    
    *   Bypasses disk serialization entirely.
        
    *   Maps weights directly via **TorchTPU runtime memory handles** into vLLM-TorchTPU (for ultra-fast RL rollouts) and SGLang-TorchTPU (for structured speculative decoding and production API serving).
        

5\. Key Features & Innovations
------------------------------

| Feature | Subsystem | Description |
|---|---|---|
| **Zero-Copy Weight Sync** | TorchTPU Runtime | Maps shared memory tensors across JAX and PyTorch runtimes on TPU v6e/v5p slices without host-disk roundtrips. |
| **Optimized Post-Training** | Tunix + MaxText | Scales distributed reinforcement learning loops (GRPO) natively on OpenXLA substrates. |
| **High-Throughput Rollouts** | vLLM-TorchTPU | Accelerates generation passes during RL training loops using paged attention. |
| **Structured Serving** | SGLang-TorchTPU | Enables advanced speculative decoding and structured API generation directly on TPU topology. |

6\. Business & Operational Benefits
-----------------------------------

*   **Maximized TPU Utilization:** Eliminates hardware idle time spent waiting for checkpoint serialization, dramatically improving total cluster ROI.
    
*   **Hyper-Fast RL Iteration Cycles:** Async rollouts allow reinforcement learning loops to ingest updated weights instantly via shared memory handles, accelerating experimentation velocity.
    
*   **Production-Ready Serving Flexibility:** Keeps training tightly coupled with industry-standard inference engines (vLLM, SGLang) without sacrificing hardware-level optimization.
    

7\. Getting Started
-------------------

### Prerequisites

*   Google Cloud TPU v6e or v5p Pod slice configured with TPU VM software version tpu-ubuntu2404-base (or equivalent).
    
*   Python 3.10+
    
*   JAX (with Cloud TPU support) & PyTorch with torch-tpu bindings.
    

### Installation & Bridge Initialization

```bash
# Spin up a TPU VM (recommended topology: v6e-8 or v5p-8)
gcloud compute tpus tpu-vm create tpu-cognition-node \
  --zone=us-central1-a \
  --accelerator-type=v6e-8 \
  --version=tpu-vm-v2-base

# Clone the repository
git clone https://github.com/jmanueldev/tpu-cognitionmesh.git
cd tpu-cognitionmesh

# Install dependencies for training and rollout planes
pip install --editable .

# Initialize the zero-copy shared memory bridge on TPU pod
python -m cognitionmesh.bridge --tpu-topology=v6e-256 --mode=initialize

bash scripts/run_hybrid_pipeline.sh

# How to Deploy the Alert Rules
# Create or Update the ConfigMap in Kubernetes:

kubectl create configmap prometheus-tpu-rules \
  --from-file=deploy/prometheus-tpu-alerts.yaml \
  -n tpu-serving-system \
  --dry-run=client -o yaml | kubectl apply -f -
# Ensure Prometheus Operator Picks Up the Rules:
# If you are using the Prometheus Operator (PrometheusRule Custom Resource), # wrap the rules group inside a PrometheusRule resource:

YAML
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: tpu-cognition-mesh-rules
  namespace: tpu-serving-system
spec:
  groups:
    - name: tpu_cognition_mesh_critical_alerts
      rules:
        # (Paste the rules list from above here)
```

8\. License
-----------

Distributed under the Apache 2.0 License. See LICENSE for more information.

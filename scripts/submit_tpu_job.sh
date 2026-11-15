#!/bin/bash
#SBATCH --job-name=tpu-cognition-mesh-v5p
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=tpu:8
#SBATCH --time=24:00:00
#SBATCH --output=logs/tpu_mesh_%j.out
#SBATCH --error=logs/tpu_mesh_%j.err

set -exo pipefail

echo "=== [Slurm] Initializing TPU v5p Multi-Host Environment ==="

# 1. Extract cluster master node info
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
master_node=${nodes[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname -I | awk '{print $1}')
export MASTER_PORT="12355"

echo "Master Node IP: $MASTER_ADDR:$MASTER_PORT"
echo "Assigned Nodes: ${nodes[@]}"

# 2. Inject PJRT and High-Bandwidth Interconnect (ICI) Networking Flags
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TF_XLA_FLAGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_megascale_auto_sharding=true"
export GRPC_POLL_STRATEGY=epoll1
export GRPC_ARG_MAX_RECV_MSG_LENGTH=134217728  # 128MB
export GRPC_ARG_MAX_SEND_MSG_LENGTH=134217728

# 3. Activate Virtual Environment / Dependencies
source /opt/venv/tpu-mesh/bin/activate

# 4. Launch Distributed Hybrid Mesh via srun across all nodes
srun --unbuffered python3 -m mesh_fabric.ray_tpu_cluster \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --world_size=$((SLURM_JOB_NUM_NODES * 8)) \
    --host_id=$SLURM_NODEID
#!/bin/bash

#SBATCH --job-name=gpt-semi
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=00:20:00
# Single environment line (local container that worked before)
#SBATCH --environment=/iopsstor/scratch/cscs/ooikonomou/ngc_pt_jan.toml
#SBATCH --output=/users/ooikonomou/Megatron-LM-new-attention/gpt-semi-%j.out
#SBATCH --error=/users/ooikonomou/Megatron-LM-new-attention/gpt-semi-%j.err
#SBATCH --no-requeue

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# torch.distributed env for env:// rendezvous
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Pick a per-job port to avoid EADDRINUSE collisions
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export WORLD_SIZE=$SLURM_NPROCS

# Minimal env (avoid UCX tweaks)
export NCCL_DEBUG=INFO

# Always run from the scratch clone inside the container
SCRATCH_DIR="/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention"
cd "$SCRATCH_DIR"

CMD="
cd \"$SCRATCH_DIR\" && \
python3 pretrain_gpt.py \
  --transformer-impl local \
  --distributed-backend gloo \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 2 \
  --hidden-size 512 \
  --num-attention-heads 8 \
  --seq-length 128 \
  --max-position-embeddings 128 \
  --micro-batch-size 2 \
  --global-batch-size 2 \
  --train-iters 10 \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --log-interval 1 \
  --eval-interval 1000000 \
  --save-interval 0 \
  --use-cpu-initialization \
  --no-masked-softmax-fusion \
  --no-gradient-accumulation-fusion \
  --no-async-tensor-model-parallel-allreduce \
  --tokenizer-type NullTokenizer \
  --vocab-size 50257 \
  --mock-data \
  --spec megatron.core.models.gpt.gpt_layer_specs get_gpt_layer_semigroup_spec
"

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} bash --norc --noprofile -c "$CMD"

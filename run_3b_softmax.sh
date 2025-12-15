#!/bin/bash
#SBATCH --job-name=3b-softmax
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --account=large-sc-2
#SBATCH --environment=/iopsstor/scratch/cscs/ooikonomou/ngc_pt_jan.toml
#SBATCH --output=/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention/logs/3b-softmax-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention/logs/3b-softmax-%j.err

echo "===== 3B SOFTMAX BASELINE ====="

MEGATRON_DIR=/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention
DATA_PATH=/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-filterrobots-merge
TOKENIZER=alehc/swissai-tokenizer
LOG_DIR=$MEGATRON_DIR/logs/3b-softmax-$SLURM_JOB_ID

mkdir -p $LOG_DIR

export PYTHONPATH=$MEGATRON_DIR
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn0,hsn1,eth0
export UCX_NET_DEVICES=all
export UCX_TLS=tcp,cuda_copy,cuda_ipc
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd $MEGATRON_DIR
pip install transformers --quiet --break-system-packages

torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    --num-layers 28 \
    --hidden-size 3072 \
    --num-attention-heads 24 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --ffn-hidden-size 8192 \
    --transformer-impl local \
    --spec megatron.core.models.gpt.gpt_layer_specs get_gpt_layer_local_spec \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --train-iters 500 \
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --min-lr 1.5e-5 \
    --lr-warmup-iters 50 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --data-path $DATA_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER \
    --split 100,0,0 \
    --log-interval 10 \
    --log-throughput \
    --tensorboard-dir $LOG_DIR/tensorboard \
    --eval-iters 0

echo "===== DONE: Softmax ====="

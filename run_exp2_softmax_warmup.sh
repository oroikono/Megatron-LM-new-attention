#!/bin/bash
#SBATCH --job-name=exp2-softmax-warmup
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --account=large-sc-2
#SBATCH --environment=/iopsstor/scratch/cscs/ooikonomou/ngc_pt_jan.toml
#SBATCH --output=/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention-wt/logs/exp2-softmax-warmup-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention-wt/logs/exp2-softmax-warmup-%j.err

echo "===== EXPERIMENT 2: SOFTMAX + WARMUP ====="
echo "Purpose: Fair convergence comparison with proper warmup"

MEGATRON_DIR=/iopsstor/scratch/cscs/ooikonomou/Megatron-LM-new-attention-wt
DATA_PATH=/users/ooikonomou/scratch/Megatron-LM-new-attention/datasets/train_data_meg_text_document
TOKENIZER=alehc/swissai-tokenizer
CKPT_DIR=$MEGATRON_DIR/checkpoints/exp2-softmax-warmup
LOG_DIR=$MEGATRON_DIR/logs/exp2-softmax-warmup-$SLURM_JOB_ID
TRIGGER_PATH=$LOG_DIR

mkdir -p $LOG_DIR $CKPT_DIR

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
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --bf16 \
    --train-iters 60000 \
    --lr 2.0e-4 \
    --lr-decay-style cosine \
    --min-lr 2.0e-5 \
    --lr-warmup-iters 2000 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --data-path $DATA_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER \
    --split 100,0,0 \
    --log-interval 10 \
    --log-throughput \
    --tensorboard-dir $LOG_DIR/tensorboard \
    --trigger-path $TRIGGER_PATH \
    --load $CKPT_DIR \
    --save $CKPT_DIR \
    --save-interval 5000 \
    --eval-iters 0

echo "===== DONE: Exp2 Softmax Warmup ====="

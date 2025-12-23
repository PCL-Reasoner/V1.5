#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=8
PP=4
MBS=1
GBS=128
SEQ_LEN=51200
# SEQ_LEN=65536
# SEQ_LEN=4096
TRAIN_ITERS=1062  # 5 epoches: 27174 * 5 / 128
NUM_LAYERS=64

DISTRIBUTED_ARGS="
    --nproc_per_node ${DEVICES_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

MEMORY_ARGS="
    --swap-attention \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 16 \
    --recompute-norm \
    --recompute-activation-function \
    --swap-optimizer
"

GPT_ARGS="    
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-distributed-optimizer \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --sequence-parallel \
    --num-layers 64 \
    --hidden-size 5120 \
    --ffn-hidden-size 27648 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 131072 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --train-iters ${TRAIN_ITERS} \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --weight-decay 0.10 \
    --lr-warmup-fraction 0.0 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --swap-attention \
    --overlap-grad-reduce \
    --overlap-param-gather
"
# --use-rotary-position-embeddings \
# --reuse-fp32-param \

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --prompt-type empty \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval ${TRAIN_ITERS} \
    --log-throughput \
    --eval-iters 0
"

RL_ARGS="
    --finetune \
    --stage opg
"


# ========================
# 启动训练命令
# ========================
torchrun ${DISTRIBUTED_ARGS} posttrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${RL_ARGS} \
    ${MEMORY_ARGS} \
    --distributed-backend nccl \
    --load "${CKPT_LOAD_DIR}" \
    --save "${CKPT_SAVE_DIR}"

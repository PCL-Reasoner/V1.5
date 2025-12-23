#!/bin/bash

# ========================
# 环境变量配置（请根据实际环境修改）
# ========================

TP=8
PP=4
CP=2
CP_TYPE='megatron_cp_algo'
MBS=1
GBS=32
SEQ_LEN=65536
TRAIN_ITERS=2000
NUM_LAYERS=64

# 分布式训练相关参数
DISTRIBUTED_ARGS="
    --nproc_per_node ${DEVICES_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

CP_ARGS="
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} 
"

MEMORY_ARGS="
    --swap-attention \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 12 \
    --recompute-norm \
    --recompute-activation-function
"

LORA_ARGS="
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"


GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-distributed-optimizer \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --sequence-parallel \
    --reuse-fp32-param \
    --num-layers ${NUM_LAYERS} \
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
    --use-flash-attn \
    --swiglu \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 5e-6 \
    --min-lr 1.0e-7 \
    --weight-decay 0.10 \
    --lr-warmup-fraction 0.05 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --seed 42 \
    --no-load-optim \
    --no-load-rng \
    --bf16 
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --prompt-type empty \
    --is-instruction-dataset \
    --variable-seq-lengths
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval $(( ${TRAIN_ITERS:-0} / 10 )) \
    --eval-interval ${TRAIN_ITERS} \
    --log-throughput \
    --eval-iters 0
"

RL_ARGS="
    --finetune \
    --stage simpo \
    --simpo-beta 2.5 \
    --gamma-beta-ratio 1.4 \
    --simpo-loss-type sigmoid \
    --is-pairwise-dataset \
    --is-instruction-dataset
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
    ${LORA_ARGS} \
    --distributed-backend nccl \
    --load "${CKPT_LOAD_DIR}" \
    --save "${LORA_CKPT_DIR}"
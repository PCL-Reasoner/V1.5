#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=4
PP=2
MBS=1
GBS=4
SEQ_LEN=16384
# SEQ_LEN=65536
TRAIN_ITERS=5000

DEVICES_PER_NODE=8
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500

DATA_PATH="/home/yaolu/workspace/Datasets/mc/prompt_cot_2_opg/"
# DATA_PATH="/home/yaolu/workspace/Datasets/mc/orca_pairwise/"
# CKPT_LOAD_DIR="/home/yaolu/workspace/Models/mcore/Qwen2.5-7B-tp4-pp1"
CKPT_LOAD_DIR="/home/fdd/workspace/models/SFT_models/Qwen2.5_7b_SFT_with_R10528/mcore_tp4_pp2"
CKPT_SAVE_DIR="./ckpt/opg_Qwen2.5-7B/"
# DATA_PATH="/home/yaolu/workspace/Datasets/mc/qwen3_cot/qwen3_cot"
# DATA_PATH="/home/yaolu/workspace/Datasets/mc/prompt_cot_2_sft"
TOKENIZER_PATH="/pcl_shared_dpc/hfhub/models/Qwen/Qwen2.5-7B/"
#TOKENIZER_PATH="/home/yaolu/workspace/Models/Qwen2.5_7b_SFT_with_R1_0528"
#CKPT_LOAD_DIR="/home/yaolu/workspace/Models/Qwen2.5_7b_SFT_with_R1_0528/mcore_tp4_pp2"
#CKPT_SAVE_DIR="/home/yaolu/workspace/Models/Qwen2.5_7b_SFT_with_R1_0528/simpo/mcore_tp4_pp2"

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
    --recompute-num-layers 14 \
    --recompute-norm \
    --recompute-activation-function
"

GPT_ARGS="    
    --sequence-parallel \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28  \
    --hidden-size 3584  \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28  \
    --max-position-embeddings 131072 \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --train-iters ${TRAIN_ITERS} \
    --lr 1.0e-6 \
    --lr-decay-style cosine \
    --min-lr 1.0e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --prompt-type empty \
    --no-pad-to-seq-lengths \
    --is-instruction-dataset
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval $(( ${TRAIN_ITERS:-0} / 1 )) \
    --eval-interval ${TRAIN_ITERS} \
    --log-throughput \
    --eval-iters 0
"

RL_ARGS="
    --finetune \
    --stage gspo \  
    --is-pairwise-dataset
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
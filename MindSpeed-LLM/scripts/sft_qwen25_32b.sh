#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# NPUS_PER_NODE=8
# MASTER_ADDR=localhost
# MASTER_PORT=6000
# NNODES=4
# NODE_RANK=1
# WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# # please fill these path configurations
# CKPT_LOAD_DIR="your model ckpt path"
# CKPT_SAVE_DIR="your model save ckpt path"
# DATA_PATH="your data path"
# TOKENIZER_PATH="your tokenizer path"

TP=8
PP=4
SEQ_LEN=65536
MBS=1
GBS=128
TRAIN_ITERS=3592 # total samples: 22990 ; 20 epoch
SAVE_ITERS=180 # 1 epoch

# MEMORY_ARGS=""
MEMORY_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 16 \
    --recompute-norm \
    --recompute-activation-function \
"

DISTRIBUTED_ARGS="
    --nproc_per_node ${DEVICES_PER_NODE} \
    --nnodes ${NUM_NODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type empty \
    --padded-samples \
    --variable-seq-lengths \
"

    # --reuse-fp32-param \ # for fp16, do not use --reuse-fp32-param

GPT_ARGS="
    --sequence-parallel \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 64  \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --hidden-size 5120  \
    --ffn-hidden-size 27648 \
    --num-attention-heads 40  \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 8 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
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
    --lr 1.25e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --fp16
"

    # --fp16 \
    # --lm-head-fp32

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
"

    # --log-throughput 
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    $MEMORY_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/tune_mcore_qwen25_32b_full.log

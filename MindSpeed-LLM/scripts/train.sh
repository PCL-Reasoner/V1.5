#!/bin/bash

# set -x

export HCCL_CONNECT_TIMEOUT=3000
export CKPT_LOAD_DIR=$1
export CKPT_SAVE_DIR=$2
export DATA_PATH=$3
export TOKENIZER_PATH=$4
export NPUS_PER_NODE=8
export MASTER_ADDR=$5
export MASTER_PORT=$6
export NNODES=$7
export NODE_RANK=$8
export WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
export GLOO_SOCKET_IFNAME=enp66s0f5
export HCCL_SOCKET_IFNAME=enp66s0f5

nohup bash scripts/tune_qwen25_32b_64k_full_ptd.sh > ${CKPT_SAVE_DIR}/node${NODE_RANK}.log 2>&1 &

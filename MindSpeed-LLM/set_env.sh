#!/bin/bash

unset PYTHONPATH
#
# export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
# export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4:$LD_PRELOAD

# export BASE_SHARE_WORK=/home/ma-user/work
export HF_ENDPOINT=https://hf-mirror.com
# source $BASE_SHARE_WORK/install/miniconda3/bin/activate
source /home/yaolu/workspace/miniconda3/bin/activate

# cann 相关环境
install_path=/home/yaolu/workspace/CANN8.3.RC1
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh
# source $install_path/mindie/set_env.sh

# mindie-llm 相关
# source /XXX/output/set_env.sh
# export PYTHONPATH=/XXX/MindIE-LLM:XXX/MindIE-LLM/examples/atb_models:$PYTHONPATH

# 日志相关
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCNED_GLOBAL_LOG_LEVEL=3
export MINDIE_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_STDOUT=1

# Torch 相关
export ASCEND_LAUNCH_BLOCKING=0
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
# 以支持torch2.5以上版本
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_DISABLE=1

# ATB 相关
export ATB_LLM_BENCHMARK_ENABLE=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=false
export ATB_LLM_LCOC_ENABLE=false

# vllm相关
export VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000000
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# HCCL相关
export HCCL_BUFFSIZE=512
export HCCL_DETERMINISTIC=true
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_CONNECT_TIMEOUT=600

# ray 相关
export RAY_DEDUP_LOGS=0
export GLOO_SOCKET_IFNAME=enp66s0f5
export HCCL_SOCKET_IFNAME=enp66s0f5

# wandb
# export WANDB_API_KEY="07f22eab1ffd7daf195ad2f4ab0f289f8be70046"

# PYTHONPATH 环境变量
if [ -z "$CURR_PROJECT_PATH" ]; then
    export CURR_PROJECT_PATH=$(pwd)
    echo "CURR_PROJECT_PATH was not set, setting it to current directory: $CURR_PROJECT_PATH"
else
    echo "CURR_PROJECT_PATH: $CURR_PROJECT_PATH"
fi

export CURR_PROJECT_PATH=/home/yaolu/workspace
export PYTHONPATH="${CURR_PROJECT_PATH}:${CURR_PROJECT_PATH}/MindSpeed:${CURR_PROJECT_PATH}/vllm-ascend:${CURR_PROJECT_PATH}/Megatron-LM:$PYTHONPATH"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

# 激活conda环境
# conda activate py311_vllm_ascend
conda activate fdd

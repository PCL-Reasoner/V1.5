#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./dataset

python ./preprocess_data.py \
    --input /home/yaolu/workspace/Datasets/merged_skywork_R10528_nvidia_57K/remove_all_correct_wrong.jsonl \
    --tokenizer-name-or-path /home/yaolu/workspace/Models/hf/Qwen2.5-32B \
    --output-prefix /home/yaolu/workspace/Datasets/merged_skywork_R10528_nvidia_57K/mcore \
    --workers 64 \
    --n-subs 8 \
    --log-interval 100 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type empty  \
    --seq-length 65536 \
    --cache-dir /home/yaolu/workspace/tmp \
    --map-keys '{"prompt":"prompt", "query":"input", "response":"response"}' # 默认值，可不传

    # --pack \
    # --neat-pack \

# --map-keys '{"prompt":"prompt","query":"input","response":"answer"}' # 默认值，可不传
# --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传

python ./preprocess_data.py \
    --input /home/yaolu/workspace/Datasets/opg_nv_community/opg_train_remove_all_wrong_cot_lt_48k.jsonl \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast \
    --tokenizer-name-or-path /pcl_shared_dpc/hfhub/models/Qwen/Qwen2.5-32B \
    --output-prefix /home/yaolu/workspace/Datasets/opg_nv_community/mc_lt_48k/ \
    --workers 64 \
    --log-interval 1000 \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type empty \
    --cache-dir /home/yaolu/workspace/tmp/ \
    --map-keys '{"prompt":"prompt", "query":"", "response":"response", "reward":"reward"}'

# --output-prefix /home/yaolu/workspace/Datasets/opg_nv_community/mc/ \
# --handler-name AlpacaStylePairwiseHandler \
# --input /home/yaolu/workspace/Datasets/opg_nv_community/merged_ds_filted_with_acc.jsonl \
# --input /home/yaolu/workspace/Datasets/opg_nv_community/merge_64_train.jsonl \
# --input /home/yaolu/workspace/Datasets/opg_nv_community/opg_train_remove_all_wrong.jsonl \
# --output-prefix /home/yaolu/workspace/Datasets/opg_nv_community/mc_remove_all_wrong/ \
# --input /home/yaolu/workspace/Datasets/opg_nv_community/merge_64_train.jsonl \
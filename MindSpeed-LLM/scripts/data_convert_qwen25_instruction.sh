# INT_PATH="/home/yaolu/workspace/Datasets/hf/qwen3_cot/qwen3_prompt_answer_cot_random_zero_one.jsonl"
INT_PATH="/home/yaolu/workspace/Datasets/prompt_cot_2/prompt_cot_2_sft.jsonl"

python ./preprocess_data.py \
    --input $INT_PATH \
    --tokenizer-name-or-path /pcl_shared_dpc/hfhub/models/Qwen/Qwen2.5-7B/ \
    --output-prefix /home/yaolu/workspace/Datasets/mc/prompt_cot_2_sft/ \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"prompt":"prompt","query":"","response":"answer"}'


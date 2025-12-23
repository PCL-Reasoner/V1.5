# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOAD_CKPT_PATH=/home/yaolu/workspace/Models/hf/Qwen2.5-32B/
SAVE_CKPT_PATH=/home/yaolu/workspace/Models/mc/Qwen2.5-32B-tp8-pp1/

# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 8 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir $LOAD_CKPT_PATH \
       --save-dir $SAVE_CKPT_PATH \
       --tokenizer-model /home/yaolu/Models/hf/Qwen2.5-32B/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh


# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --load-dir /home/yaolu/workspace/MindSpeed-LLM/work_dir/opg_qwen25_32b_cot_lt_48k/model_ckpt/ \
    --save-dir /home/yaolu/workspace/Models/PCL-Reasoner-V1/

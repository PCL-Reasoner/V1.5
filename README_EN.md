# **PCL-Reasoner-V1.5 Mathematical Reasoning Model**

## Model Overview

PCL-Reasoner-V1.5 is a 32B-parameter large language model specifically designed for mathematical reasoning. Built upon Qwen2.5-32B-Base, the model is trained using Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). A key innovation in our approach is the adoption of **Offline Reinforcement Learning (Offline RL)**, which significantly improves training efficiency compared to traditional online RL methods.

On public benchmarks, PCL-Reasoner-V1.5 achieves state-of-the-art performance among 32B-scale models:

- **91.3%** average accuracy on the AIME 2024 benchmark  
- **91.0%** average accuracy on the AIME 2025 benchmark  

All experiments were conducted exclusively on Huawei Ascend NPUs using publicly available datasets.

To foster open collaboration and practical application, we have fully open-sourced the model weights, data processing pipelines, and training code for PCL-Reasoner-V1.5. This model not only represents one of the leading 32B mathematical reasoning models available today but also provides developers with valuable hands-on experience in domain-specific offline reinforcement learning and post-training methodologies. Users can easily deploy and evaluate the model by following the instructions below to explore advanced post-training techniques!

---

## Development Guide

### 1. Training Code

PCL-Reasoner-V1.5 is fine-tuned from PCL-Reasoner-V1 using the MindSpeed-LLM framework. We introduced a new trainer module `opg_trainer.py` and enhanced the dataset preprocessing pipeline to include a `reward` field. To facilitate reproducibility within the open-source community, we have packaged the complete training code under the `MindSpeed-LLM` directory in this repository.

### 2. Environment and Data Preparation

#### 2.1 Environment Setup

| Software      | Version        |
|---------------|----------------|
| Firmware & Driver | 24.1.rc3.5 |
| CANN          | 8.3.RC1        |
| Python        | 3.10           |
| vllm-ascend   | 0.9.1          |
| MindSpeed-LLM     | commit: 887c2d868 |

#### 2.2 Data Processing

##### 2.2.1 Dataset Download

In prior work, we had already elevated PCL-Reasoner-V1’s mathematical reasoning capability to a high level. To further refine performance, we selected challenging problems from NVIDIA’s publicly released **Nemotron-Post-Training-Dataset-v1** for subsequent reinforcement training.

| Dataset Name                    | Link                                                                                                                     |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| nvidia/Nemotron-Post-Training-Dataset-v1 | [https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) |

##### 2.2.2 Data Preprocessing

The downloaded dataset is in Parquet format. We first convert it to JSONL for easier handling:

```bash
# dir_path_to_parquet_files=Nemotron-Post-Training-Dataset-v1/data
# output_dir_path=Nemotron-Post-Training-Dataset-v1/orig2jsonl
cd data_preprocess
python convert_parquet2jsonl.py $dir_path_to_parquet_files $output_dir_path --workers 128
# Merge all JSONL files into one
cat $output_dir_path/*jsonl > Nemotron-Post-Training-Dataset-v1/all_samples.jsonl
```

Statistical analysis revealed that each problem in the dataset was sampled multiple times, with only correct Chain-of-Thought (CoT) responses retained. Based on this, we computed per-problem accuracy and CoT length. The preprocessing consists of three steps:

1. **Separate Fully Correct vs. Partially Correct Samples**:
    Most problems have 1–16 CoTs; a few have 17–32, likely due to duplication. We infer that the original dataset performed 16 inference attempts per problem and kept only correct CoTs. Thus:
    - Filter out samples with exactly 16 or 32 CoTs (fully correct)
    - Retain partially correct samples (1–15 CoTs)
    ```bash
    # cd data_preprocess 
    python split_all_right_and_partial_right.py all_samples.jsonl \
    --complete_output all_right_samples.jsonl \
    --incomplete_output partial_right_samples.jsonl \
    --processes 128
    ```
    From 2,044,407 total CoTs, we obtained:
    - 1,189,392 fully correct CoTs (filtered out)
    - 855,015 partially correct CoTs (retained)

2. **Filter Long CoT Samples**:
    From the 855K partially correct CoTs, select those with average CoT length > 32K tokens:
    ```bash
    # cd data_preprocess 
    python static_and_filter_cot.py partial_right_samples.jsonl partial_right_samples_cot_filter.jsonl      path_to_tokenizer --processes 128
    ```
    Result: ~34K problems with CoT > 32K tokens.

3. **Extract Unique Problems**:
    Deduplicate to retain only the first occurrence of each problem:
    ```bash
    # cd data_preprocess 
    python extract_first_problem.py partial_right_samples_cot_filter.jsonl partial_right_problem.jsonl
    ```
    Final dataset: ~6K unique problems.

##### 2.2.3 Model Sampling

Using the **PCL-Reasoner-V1** model, we sampled 8 responses per problem (total: 48K CoTs) with the following configuration (identical to evaluation settings):

| Sampling Hyperparameter       | Value                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top_k         | 40                                         |
| top_p         | 0.95                                       |
| max_model_len    | 131072                                     |
| system_prompt_type | amthinking |

##### 2.2.4 CoT Correctness Evaluation

Traditional rule-based evaluators (e.g., math_verify) often fail on complex math problems. Instead, we employed **Qwen3-32B** as an evaluator with a specialized prompt to judge whether the final answer in a CoT matches the ground truth.

**Evaluation Prompt Template**:

```text
As a math scoring expert, given a standard answer, and a candidate answer, you need to compare whether the standard answer and the candidate answer are consistent. If they are consistent, return 1; if not, return 0. Remember the returned value should always be put in the \boxed{}.
Here are a few points to note:
1. For the candidate answer, only consider the content inside \boxed{}, ignoring any other text or error. If no \boxed{} found, return 0 directly.
2. If the standard answer and the candidate answer are different but mathematically equivalent, return 1.
3. For answers involving decimals, if most digits are the same and only the last one or two digits differ, you may considerably return 1.
4. For all other cases where the standard answer and the candidate answer do not match, return 0.
Here is a task example:
<Standard Answer Begin>
{Standard answer}
<Standard Answer End>
<Candidate Answer Begin>
{Candidate answer}
<Candidate Answer End>
Please put your return value (0 or 1) as required above in the \boxed{} without any explanation or description.
```

After evaluation:
- 22,990 positive samples
- 25,522 negative samples

#### 2.3 Model Weight Preparation

Download the base model **PCL-Reasoner-V1** from Hugging Face:

| **Model Name**          | **Link**                                                                           |
| ----------------- | ---------------------------------------------------------------------------------- |
| PCL-Reasoner-V1 | [https://huggingface.co/PCL-Reasoner/V1](https://huggingface.co/PCL-Reasoner/V1) |

### 3. **Training Procedure**

Training is based on the **MindSpeed-LLM** framework and includes the following steps:

#### 3.1 Model Weight Conversion

##### 3.1.1 Download Hugging Face Weights

```bash
huggingface-cli download PCL-Reasoner/V1 --local-dir ~/local/PCL-Reasoner/V1
```

##### 3.1.2 Convert to MCore Format

MindSpeed-LLM requires weights in MCore format. Convert using:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

hf_model_path=/path/to/hf/model
cd MindSpeed-LLM
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 4 \
    --add-qkv-bias \
    --load-dir $hf_model_path \
    --save-dir ${hf_model_path}/mcore_tp8_pp4/ \
    --tokenizer-model $hf_model_path/tokenizer.json \
    --model-type-hf llama2 \
    --params-dtype bf16
```

**Parameter Descriptions**:

- --use-mcore-models: Enable MCore models
- --model-type: e.g., GPT
- --load-model-type / --save-model-type: hf (Hugging Face) → mg (MCore)
- --target-tensor-parallel-size: Tensor parallelism degree
- --target-pipeline-parallel-size: Pipeline parallelism degree
- --add-qkv-bias: Add bias to QKV projections
- --params-dtype: e.g., bf16
  

#### 3.2 Dataset Conversion

Convert the 48K CoT samples (JSONL) into MindSpeed-LLM-compatible binary format:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

hf_model_path=/path/to/hf/model
python preprocess_data.py \
    --input /path/to/opg_train_remove_all_wrong_cot_lt_48k.jsonl \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast \
    --tokenizer-name-or-path $hf_model_path \
    --output-prefix /path/to/mc_lt_48k/ \
    --workers 64 \
    --log-interval 1000 \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type empty \
    --cache-dir /path/to/tmp/ \
    --map-keys '{"prompt":"prompt", "query":"", "response":"response", "reward":"reward"}'
```

#### 3.3 Training Configuration

- Global batch size: 128
- Learning rate: Cosine decay from $6×10^{−5}$ to $1×10^{−7}$ 
- Warmup ratio: 0.05
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.95$)
- Hardware: 16 × Atlas 910C SuperPoD nodes (8 chips/node)
- Epochs: 4 (~116 hours total)

**Data Packing**: Enabled during SFT to concatenate variable-length samples into fixed 48K-token sequences, eliminating padding overhead and accelerating training.

#### 3.4 Launch Training

```bash
# Step 1: Activate environment
source /path/to/set_env.sh

# Step 2: Start multi-node training
cd MindSpeed-LLM
bash scripts/launch_multi_nodes.sh node_list.txt
```

### 4. Evaluation Procedure

We use [LLMEval](https://gitee.com/jianzhnie/LLMEval) —a toolkit developed by our team for LLM reasoning evaluation—to benchmark PCL-Reasoner-V1.5. LLMEval supports both vLLM and SGLang backends and has been validated on Ascend hardware. See the LLMEval documentation for details.

#### 4.1 Evaluation Environment Setup

##### 4.1.1 Install vLLM and vLLM-Ascend

```bash
pip install vllm==0.9.1
pip install vllm-ascend==0.9.1
```
##### 4.1.2 Set Up LLMEval

```bash
git clone https://gitee.com/jianzhnie/LLMEval.git
cd LLMEval
pip install -e .
```

#### 4.2 Run Evaluation

##### Step 1: Launch vLLM Server

```bash
source set_env.sh

model_path="/path/to/pcl_reasoner_v1"
model_name="PCL-Reasoner-v1"
num_gpus=8
max_model_len=131072
gpu_memory_utilization=0.9

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len \
    --enforce-eager \
    --port 8090
```

##### Step 2: Run Inference

```bash
source set_env.sh
set -euo pipefail

output_dir="./output/PCL-Reasoner-v1"
model_name="PCL-Reasoner-v1"
base_url="http://127.0.0.1:8090/v1"
n_samples=64
mkdir -p "${output_dir}"

# AIME25
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime25.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --temperature 0.6 \
    --system_prompt_type amthinking \
    --max_workers 64

# AIME24
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime24.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --temperature 0.6 \
    --system_prompt_type amthinking \
    --max_workers 64

echo "🎉 All inference tasks completed successfully!"
```

> Note: Repeated sampling (64×) reduces evaluation variance but may take >8 hours depending on hardware.

**Evaluation Sampling Parameters**:

| Sampling Hyperparameter       | Value                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top_k         | 40                                         |
| top_p         | 0.95                                       |
| max_model_len    | 131072                                     |
| system_prompt_type | amthinking |

##### Step 3: Scoring

```bash
source set_env.sh
set -euo pipefail

output_dir="./output/PCL-Reasoner-v1"
n_samples=64
reval_dir="${output_dir}/eval_score"
mkdir -p "${reval_dir}"

# AIME24
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime24_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime24" \
    --max_workers 16 \
    > "${reval_dir}/aime24_bz${n_samples}_res_result.txt"

# AIME25
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime25_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime25" \
    --max_workers 16 \
    > "${reval_dir}/aime25_bz${n_samples}_res_result.txt"

echo "🎯 Evaluation completed successfully!"
```

#### 4.3 Evaluation Results

We report Avg@32 (average over 32 samples) for robustness:


<style>
  table { border-collapse: collapse; width: 100%; margin-left: auto;margin-right: auto;}
  th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
</style>

<!-- 表格主体 -->

<table>
  <tr>
    <th>模型规格</th>
    <th>模型</th>
    <th>AIME 24</th>
    <th>AIME 25</th>
  </tr>
  <!-- 合并行表头 >100B -->
  <tr>
    <th rowspan="6">&gt;100B</th>
  </tr>
  <!-- >100B 组数据行 -->
  <tr>
    <td>DeepSeek-R1</td>
    <td><span style="color:grey">79.8</span></td>
    <td><span style="color:grey">70</span></td>
  </tr>
  <tr>
    <td>DeepSeek-R1-0528</td>
    <td><span style="color:red">91.4</span></td>
    <td><span style="color:red">87.5</span></td>
  </tr>
  <tr>
    <td>Qwen3-235B-A22B</td>
    <td><span style="color:grey">85.7</span></td>
    <td><span style="color:grey">81.5</span></td>
  </tr>
  <tr>
    <td>OpenAI-o3</td>
    <td><span style="color:red">91.6</span></td>
    <td><span style="color:red">88.9</span></td>
  </tr>
  <tr>
    <td>Gemini-2.5-Pro-0506</td>
    <td><span style="color:red">90.8</span></td>
    <td><span style="color:grey">83</span></td>
  </tr>
  <!-- 分隔行 -->
  <tr>
    <td colspan="4"></td>
  </tr>
  <!-- 合并行表头 32B -->
  <tr>
    <th rowspan="7">32B</th>
  </tr>
  <!-- 32B 组数据行 -->
  <tr>
    <td>Qwen3-32B</td>
    <td><span style="color:grey">81.4</span></td>
    <td><span style="color:grey">72.9</span></td>
  </tr>
  <tr>
    <td>QwQ-32B</td>
    <td><span style="color:grey">79.5</span></td> 
    <td><span style="color:grey">69.5</span></td>
  </tr>
  <tr>
    <td>DeepSeek-R1-Distill-Qwen-32B</td>
    <td><span style="color:grey">72.6</span></td>
    <td><span style="color:grey">49.6</span></td> 
  </tr>
  <tr>
    <td>Skywork-OR1-32B</td>
    <td><span style="color:grey">82.2</span></td>
    <td><span style="color:grey">73.3</span></td>
  </tr>
  <tr>
    <td>AM-Thinking-v1</td>
    <td><span style="color:grey">85.3</span></td>
    <td><span style="color:grey">74.4</span></td>
  </tr>
  <tr>
    <td>PCL-Reasoner-v1</td>
    <td><p style="font-weight: bold;">85.7</p></td> 
    <td><p style="font-weight: bold;">84.2</p></td> 
  </tr>
</table>

> *(注：模型在AIME24/25评测集上的生成结果文件已同步上传至 `PCL-Reasoner-V1.5/eval_result`目录，供开发者用于模型验证与效果比对参考）*

## Ciation

```bibtex
@article{PCL-Reasoner-v1.5,
  title={PCL-Reasoner-v1.5: A Math Problem Solver with Chain of Thought Reasoning},
  author={Yao Lu, Deng Dong Fan, Jianzheng Nie, et al.},
  journal={arXiv preprint arXiv:2405.14524},
  year={2024}
}
```

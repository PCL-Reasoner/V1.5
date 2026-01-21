# **PCL-Reasoner-V1.5 Mathematical Reasoning Model**

[中文文档](README_CN.md) | English

![PCL-Reasoner-V1.5](./images/pcl_reasoner_v15.png)

## Model Overview

PCL-Reasoner-V1.5 is a 32B-parameter large language model specifically designed for mathematical reasoning. The model is built upon Qwen2.5-32B-Base and trained through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). A key innovation in our approach is the adoption of Offline RL (Offline Reinforcement Learning), which significantly enhances training efficiency compared to traditional online RL methods.

PCL-Reasoner-V1.5 demonstrates outstanding performance among 32B-scale models on public datasets:

- Achieves 90.9% average accuracy on the AIME 2024 benchmark
- Achieves 85.7% average accuracy on the AIME 2025 benchmark

All experiments were conducted on Huawei Ascend NPU using only publicly available datasets.

To promote technology sharing and application, we have fully open-sourced the PCL-Reasoner-V1.5 model weights, data processing scripts, and training code. This model represents one of the leading 32B mathematical reasoning models today and provides developers with valuable practical experience in offline reinforcement learning for specialized domains and post-training solutions. Users can easily deploy and experience the model by following the tutorials below, and explore the practical methods and techniques of post-training!


## Development Guide

### 1. Training Code

PCL-Reasoner-V1.5 is fine-tuned based on PCL-Reasoner-V1, with the training pipeline implemented on the MindSpeed-LLM framework. We primarily added `opg_trainer.py` and incorporated a `reward` keyword in dataset processing. To facilitate reproducibility for the open-source community, we have packaged the entire training code in the `MindSpeed-LLM` directory.

### 2. Environment and Data Preparation

#### 2.1 Environment Setup:

| Software      | Version                 |
| --------- | ----------        |
| Firmware & Driver | 24.1.rc3.5        |
| CANN      | 8.3.RC1           |
| Python    | 3.10              |
| vllm-ascend   | 0.9.1          |
| MindSpeed-LLM | commit: 887c2d868 |


#### 2.2 Data Processing

##### 2.2.1 Dataset Download

In our preliminary work, we had already elevated the mathematical reasoning capabilities of PCL-Reasoner-V1 to a high level. To further optimize model performance, we selected challenging problems from NVIDIA's publicly available Nemotron-Post-Training-Dataset-v1 for subsequent reinforcement training.

| Dataset Name                    | Dataset Link                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| nvidia/Nemotron-Post-Training-Dataset-v1 | [https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) |

##### 2.2.2 Data Preprocessing

After downloading, the dataset is in Parquet format. We first convert it to JSONL format for convenient subsequent processing.

```bash
# dir_path_to_parquet_files=Nemotron-Post-Training-Dataset-v1/data
# output_dir_path=Nemotron-Post-Training-Dataset-v1/orig2jsonl
cd data_preprocess
python convert_parquet2jsonl.py $dir_path_to_parquet_files $output_dir_path  --workers 128
# Merge all datasets into a single jsonl file
cat $output_dir_path/*jsonl > Nemotron-Post-Training-Dataset-v1/all_samples.jsonl
```

Through statistical analysis, we discovered that in the Nemotron-Post-Training-Dataset-v1 dataset, each problem appears multiple times, retaining only correct Chain of Thought (CoT) samples. Based on this observation, we can calculate the accuracy and CoT length for each problem. The entire data preprocessing process consists of three steps:

1. **Separating Fully Correct and Partially Correct Samples**: We counted the CoT quantities for identical problems in the original dataset, discovering that most problems correspond to 1-16 CoT samples, with very few problems corresponding to 17-32 CoT samples. Based on this, we inferred that the original dataset performed 16 inference passes on each problem and retained only correct CoT samples. Samples with 17-32 counts can be viewed as slightly duplicated problems. Therefore, the first step is to filter out samples with 16 and 32 CoT counts (i.e., fully correct samples) and retain partially correct samples:
    ```bash
    # cd data_preprocess 
    python split_all_right_and_partial_right.py all_samples.jsonl --complete_output all_right_samples.jsonl --incomplete_output partial_right_samples.jsonl --processes 128 
    ```
   The original dataset contained 2,044,407 CoT samples. After processing, we obtained 1,189,392 fully correct CoT samples (with all-correct questions filtered out) and 855,015 partially correct CoT samples.

2. **Filtering Long CoT Samples**: From the 855,015 partially correct CoT samples, we selected samples with average CoT length greater than 32K tokens:
    ```bash
    # cd data_preprocess 
    python static_and_filter_cot.py partial_right_samples.jsonl partial_right_samples_cot_filter.jsonl path_to_tokenizer --processes 128
    ```
    After processing, we obtained approximately 34K problems, each with average CoT length exceeding 32K tokens.

3. **Extracting Unique Problems**: From the 34K CoT samples, we extracted the first occurrence of each unique problem:
   
   ```bash
   # cd data_preprocess 
   python extract_first_problem.py partial_right_samples_cot_filter.jsonl partial_right_problem.jsonl
   ```
   After final processing, we obtained approximately 6K unique problems.


##### 2.2.3 Model Sampling

After obtaining the 6K dataset, we used the `PCL-Reasoner-V1` model for sampling, drawing 8 samples per problem to generate reasoning results. Sampling configuration is consistent with the evaluation settings below:

| Sampling Hyperparameter       | Value                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top_k         | 40                                         |
| top_p         | 0.95                                       |
| max_model_len    | 131072                                     |
| system_prompt_type | amthinking |

After sampling, we obtained 48K CoT samples.

##### 2.2.4 Evaluating Correctness of Sampled CoT

Based on previous training experience, we found that traditional methods using the Python `math_verify` package cannot effectively evaluate CoT answer correctness in all scenarios. For complex mathematical problems, rule-based matching approaches often produce various false judgments. Therefore, we employed the Qwen3-32B model to assess the correctness of CoT answers, with the specific approach as follows:

1. We designed specialized prompts for the `Qwen3-32B` model to determine whether the final answer contained in the CoT is consistent with the `ground truth` provided by the problem;
2. We deployed the `Qwen3-32B` model to perform inference evaluation on 48K problems;
3. We compared the answer contained in the last 300 tokens of the CoT generated by `Qwen3-32B` with the `ground truth` provided by the problem, thereby determining whether the CoT is correct.

The evaluation prompt template is as follows:

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

Finally, we obtained 22,990 positive samples and 25,522 negative samples.

#### 2.3 Model Weights Preparation

Users can download `PCL-Reasoner-V1` weights from the official HuggingFace repository:

| Model Name          | Weights Link                                                                           |
| ----------------- | ---------------------------------------------------------------------------------- |
| PCL-Reasoner-V1 | [https://huggingface.co/PCL-Reasoner/V1](https://huggingface.co/PCL-Reasoner/V1) |

### 3 Training Pipeline

Our training is based on the MindSpeed-LLM framework and includes the following steps:

#### 3.1 Model Weights Conversion

##### 3.1.1 Download HuggingFace Model Weights

Download the PCL-Reasoner/V1 model weights from HuggingFace to your local machine.

```bash
# download  model
huggingface-cli download  PCL-Reasoner/V1  --local-dir ~/local_dir/PCL-Reasoner/V1
```

##### 3.1.2 Convert Model Weights Format

The MindSpeed-LLM framework is built on MindSpeed and reads weights in mcore format. Before training, HuggingFace weights need to be converted to Mcore format. The script can be launched with bash, and configuration parameters can be adjusted according to your environment. The launch command and configuration parameters are as follows:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

hf_model_path=~/local_dir/PCL-Reasoner/V1
# Set the required parameters for weights conversion
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

###### Parameter Descriptions

- `use-mcore-models`: Enable MCore models;
- `model-type`: Specify model type, e.g., GPT;
- `load-model-type`: Specify the model type to load, e.g., hf (HuggingFace);
- `save-model-type`: Specify the model type to save, e.g., mg;
- `target-tensor-parallel-size`: Set the target tensor parallelism size;
- `target-pipeline-parallel-size`: Set the target pipeline parallelism size;
- `add-qkv-bias`: Whether to add QKV bias;
- `load-dir`: Path to load HuggingFace weights;
- `save-dir`: Path to save converted weights;
- `tokenizer-model`: Path to the tokenizer model file;
- `model-type-hf`: Specify HuggingFace model type, e.g., llama2;
- `params-dtype`: Specify parameter data type, e.g., bf16.



#### 3.2 Dataset Conversion

After inference, we obtained 48K CoT samples in JSONL format, containing the problem, inference results, and corresponding CoT for the inference results. These need to be converted to a format readable by MindSpeed-LLM:


```bash
# Please modify the set_env.sh path according to your actual environment
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

##### Parameter Descriptions

- `input`: Path to input JSONL file
- `tokenizer-type`: Type of tokenizer to use
- `tokenizer-name-or-path`: Path to tokenizer
- `output-prefix`: Path to output files
- `workers`: Number of threads for data processing
- `log-interval`: Sample interval for logging
- `handler-name`: Name of the selected `data handler`
- `prompt-type`: Type of prompt in the input, `empty` means no prompt (indicating the original prompt already contains the chat template)
- `cache-dir`: Cache directory
- `map-keys`: Field mapping for the input JSONL file


#### 3.3 Training Configuration

We adopted an inspired training strategy. Global batch size was set to 128, with learning rate decaying from $6×10^{−5}$ to $1×10^{−7}$ following a cosine schedule, and a warm-up ratio of 0.05. AdamW optimizer parameters were configured as $\beta_1=0.9$ and $\beta_2=0.95$. Training was conducted on 16 Atlas 910C SuperPoD nodes (each containing 8 accelerators). The entire fine-tuning process involved 4 epochs and took approximately 116 hours. The corresponding training loss curves are shown in Figure \ref{fig:loss}.

To maximize computational efficiency, we enabled data packing during the supervised fine-tuning phase. This feature allows concatenating samples of varying lengths within each batch into a preset sequence length (48K tokens). By merging multiple short sequences into one long sequence, we effectively eliminated redundant computation caused by sequence padding, significantly accelerating training speed.

#### 3.4 Launching Training

The training process consists of three steps:

1. Activate environment: `source /path/to/set_env.sh`
2. Launch training: `cd MindSpeed-LLM; bash scripts/lauch_multi_nodes.sh node_list.txt`


#### 3.5 Converting Model Weights to HuggingFace Format

After training is complete, weights need to be converted from Megatron-LM format to HuggingFace standard format, ensuring they can be used for continued training and inference in the HuggingFace environment. The weight conversion script is as follows:

```bash
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_weights/sft_pcl_model/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ~/local_dir/PCL-Reasoner/V1/  # <-- Please fill in the original HF model path; new weights will be saved in ./local_dir/PCL-Reasoner/V1/mg2hf
```

Note: When converting to HuggingFace weights, you must set --target-tensor-parallel-size = 1 and --target-pipeline-parallel-size = 1.

After conversion is complete, the new HuggingFace format weights will be stored in the `~/local_dir/PCL-Reasoner/V1/mg2hf` directory. You can then load and perform inference using vllm, sglang, or huggingface frameworks.


### 4. Evaluation Pipeline:

We use [LLMEval](https://gitee.com/jianzhnie/LLMEval) to evaluate the model. LLMEval is an evaluation tool developed by our team, primarily designed for evaluating large model inference. It supports both vllm and sglang inference backends and multiple evaluation datasets. It has successfully reproduced the results of multiple open-source inference models in the Ascend environment. For more details, please refer to [LLMEval Usage Tutorial](https://gitee.com/jianzhnie/LLMEval).

#### 4.1 Evaluation Environment Configuration

#### 4.1.1 Install vllm and vllm-ascend

Please refer to the [vllm documentation](https://vllm-ascend.readthedocs.io/en/latest/getting_started/installation.html) and [vllm-ascend documentation](https://vllm-ascend.readthedocs.io/en/latest/getting_started/installation.html) to install vllm and vllm-ascend environments.

```bash
# Install vllm-project/vllm from pypi
pip install vllm==0.9.1

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.9.1
```

#### 4.1.2 Configure llmeval Environment

```bash
# Clone the LLMEval repository
git clone https://gitee.com/jianzhnie/LLMEval.git

# Navigate to the LLMEval directory
cd LLMEval
# Install LLMEval in editable mode
pip install -e .
```


#### 4.2 Start Evaluation

##### Step 1: Launch vLLM Server

```bash
source set_env.sh

model_path="~/local_dir/PCL-Reasoner/V1"
model_name="PCL-Reasoner-v1"

num_gpus=8
max_model_len=131072  # ✅ Supports 128k context length
gpu_memory_utilization=0.9  # ✅ Increases memory utilization

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len  \
    --enforce-eager \
    --port 8090
```

Adjust the `tensor_parallel_size` parameter according to available devices.


##### Step 2: Submit Inference Tasks

After launching vLLM service, run the inference script to generate responses and save results to a specified output file.

```bash
source set_env.sh

set -euo pipefail

# --- Configuration ---
output_dir="./output/PCL-Reasoner-v1"
model_name="PCL-Reasoner-v1"

base_url="http://127.0.0.1:8090/v1"
n_samples=64  # Default sample size for aime24 and aime25

# Create output directory if it doesn't exist
mkdir -p "${output_dir}"

# --- Run Inference Tasks ---
# aime25 (repeated sample 64 times)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime25.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --temperature 0.6  \
    --system_prompt_type amthinking \
    --max_workers 64

# aime24 (repeated sample 64 times)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime24.jsonl" \
    --input_key "prompt" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --temperature 0.6  \
    --system_prompt_type amthinking \
    --max_workers 64

echo "🎉 All inference tasks completed successfully!"
```

**Note:** We use repeated sampling to reduce evaluation variance, but this may take a long time to complete (potentially over 8 hours depending on hardware).


The evaluation hyperparameters we used are shown in the table below:

| Sampling Hyperparameter       | Value                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top_k         | 40                                         |
| top_p         | 0.95                                       |
| max_model_len    | 131072                                     |
| system_prompt_type | amthinking |

##### Step 3: Evaluation Scoring

After completing inference, use the following command to perform evaluation:

```bash
source set_env.sh

set -euo pipefail

# --- Configuration ---
output_dir="./output/PCL-Reasoner-v1"
n_samples=64 # Default sample size for aime24 and aime25

# Evaluation output directory
reval_dir="${output_dir}/eval_score"

# Create evaluation directory if it doesn't exist
mkdir -p "${reval_dir}"

# --- Evaluate Each Task ---
# Evaluate aime24
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime24_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime24" \
    --max_workers 16 \
    > "${reval_dir}/aime24_bz${n_samples}_res_result.txt"

# Evaluate aime25
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime25_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime25" \
    --max_workers 16 \
    > "${reval_dir}/aime25_bz${n_samples}_res_result.txt"

echo "🎯 Evaluation completed successfully!"
```


####  4.3 Evaluation Results

Detailed evaluation results on AIME24/AIME25 are shown in the table below. To ensure evaluation accuracy, we used the Avg@32 metric (average of 32 samples) for our evaluation:

<!-- Table base styling (optional) -->
<!-- Table content -->

<div class="table-wrapper">
<table>
  <thead>
    <tr>
      <th>Model Scale</th>
      <th>Model</th>
      <th>AIME 24</th>
      <th>AIME 25</th>
    </tr>
  </thead>
  <tbody>
    <!-- Merged row header &gt;100B -->
    <tr>
      <th class="row-label" rowspan="6" scope="rowgroup">&gt;100B</th>
    </tr>
    <tr>
      <td>DeepSeek-R1</td>
      <td class="score"> <span style="color: grey;">79.8</td>
      <td class="score"> <span style="color: grey;">70</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-0528</td>
      <td class="score--high"><span style="color: red; font-weight: bold;">91.4</td>
      <td class="score--high"><span style="color: red; font-weight: bold;">87.5</td>
    </tr>
    <tr>
      <td>Qwen3-235B-A22B</td>
      <td class="score"><span style="color: grey;">85.7</td>
      <td class="score"><span style="color: grey;">81.5</td>
    </tr>
    <tr>
      <td>OpenAI-o3</td>
      <td class="score--high"><span style="color: red; font-weight: bold;">91.6</td>
      <td class="score--high"><span style="color: red; font-weight: bold;">88.9</td>
    </tr>
    <tr>
      <td>Gemini-2.5-Pro-0506</td>
      <td class="score--high"><span style="color: red; font-weight: bold;">90.8</td>
      <td class="score">83</td>
    </tr>
    <!-- Separator row -->
    <tr class="separator">
      <td colspan="4"></td>
    </tr>
    <!-- Merged row header 32B -->
    <tr>
      <th class="row-label" rowspan="9" scope="rowgroup">32B</th>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td class="score"><span style="color: grey;">81.4</td>
      <td class="score"><span style="color: grey;">72.9</td>
    </tr>
    <tr>
      <td>QwQ-32B</td>
      <td class="score"><span style="color: grey;">79.5</td>
      <td class="score"><span style="color: grey;">69.5</td>
    </tr>
    <tr>
      <td>DeepSeek-R1-Distill-Qwen-32B</td>
      <td class="score"><span style="color: grey;">72.6</td>
      <td class="score"><span style="color: grey;">49.6</td>
    </tr>
    <tr>
      <td>Skywork-OR1-32B</td>
      <td class="score"><span style="color: grey;">82.2</td>
      <td class="score"><span style="color: grey;">73.3</td>
    </tr>
    <tr>
      <td>AM-Thinking-v1</td>
      <td class="score"><span style="color: grey;">85.3</td>
      <td class="score"><span style="color: grey;">74.4</td>
    </tr>
    <tr>
      <td>OpenReasoning-Nemotron-32B</td>
      <td class="score"><span style="color: grey;">89.2</td>
      <td class="score"><span style="color: grey;">84.2</td>
    </tr>
    <tr>
      <td>PCL-Reasoner-v1</td>
      <td class="score"><span style="color: grey;">85.7</td>
      <td class="score"><span style="color: grey;">84.2</td>
    </tr>
    <tr>
      <td>PCL-Reasoner-v1.5</td>
      <td class="score--high"><span style="color: green; font-weight: bold;">90.9</td>
      <td class="score--high"><span style="color: green; font-weight: bold;">85.7</td>
    </tr>
  </tbody>
</table>
</div>

> *(Note: The model's generated result files on the AIME24/25 evaluation sets have been synchronized and uploaded to the `PCL-Reasoner-V1.5/eval_result` directory for developers to use in model validation and effect comparison)*

## Citation

```bibtex
@article{PCL-Reasoner-v1.5,
  title={PCL-Reasoner-v1.5: A Math Problem Solver with Chain of Thought Reasoning},
  author={Yao Lu, Deng Dong Fan, Jianzheng Nie, et al.},
  journal={arXiv preprint arXiv:2405.14524},
  year={2026}
}
```

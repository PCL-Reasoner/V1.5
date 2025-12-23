# **PCL-Reasoner-V1.5 数学推理模型**

## 模型概览

PCL-Reasoner-V1.5 是一个专为数学推理设计的 32B 参数大语言模型。该模型基于 Qwen2.5-32B-Base 构建，通过监督微调（Supervised Fine-Tuning, SFT）与强化学习（Reinforcement Learning, RL）进行训练。我们方法的一项关键创新在于采用了离线强化学习（Offline RL），相较于传统的在线强化学习方法，显著提升了训练效率。
在公开数据集上，PCL-Reasoner-V1.5 在 32B 规模模型中表现卓越：

- 在 AIME 2024 基准测试中达到 91.3% 的平均准确率
- 在 AIME 2025 基准测试中达到 91.0% 的平均准确率

所有实验均在华为昇腾（Ascend）NPU 上完成，仅使用公开可用的数据集。

为促进技术共享与应用，我们已完整开源了PCL-Reasoner-V1.5的模型权重、数据处理及训练代码。该模型不仅是当下领先的32B数学推理模型之一，更为开发者提供了宝贵的专业领域离线强化学习实践经验与后训练解决方案。用户可参照以下教程轻松部署体验，深入探索后训练的实践方法与奥秘！


## 开发指导

### 1. 训练代码

PCL-Reasoner-V1.5基于PCL-Reasoner-V1进行微调后训练，训练流程基于MindSpeed-LLM框架实现，我们主要增加了一个`opg_trainer.py`，和对数据集的处理中，添加了`reward`关键字。为了方便开源社区复现，我们将整个训练代码打包放到当前目录`MindSpeed-LLM`下面。

### 2.环境及数据准备

#### 2.1 安装环境：

| 软件      | 版本       |
| --------- | ---------- |
| 固件&驱动 | 24.1.rc3.5 |
| CANN      | 8.3.RC1    |
| Python    | 3.10       |
| MindSpeed | commit: 887c2d868 |


#### 2.2 数据处理

##### 2.2.1 数据集下载

在前期工作中，我们已经将PCL-Reasoner-V1的数学推理能力提升到了很高的水平。为了进一步优化模型性能，我们考虑从NVIDIA公开的Nemotron-Post-Training-Dataset-v1数据集中筛选具有一定难度的题目，用于后续的强化训练。

| 数据集名称                    | 数据集链接                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| nvidia/Nemotron-Post-Training-Dataset-v1 | [https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) |

##### 2.2.2 数据预处理

数据集下载后为parquet格式，我们首先将数据集转换为jsonl格式，方便后续处理。

```bash
# dir_path_to_parquet_files=Nemotron-Post-Training-Dataset-v1/data
# output_dir_path=Nemotron-Post-Training-Dataset-v1/orig2jsonl
cd data_preprocess
python convert_parquet2jsonl.py $dir_path_to_parquet_files $output_dir_path  --workers 128
# 将数据集合并为一个jsonl文件
cat $output_dir_path/*jsonl > Nemotron-Post-Training-Dataset-v1/all_samples.jsonl
```

经过统计分析，我们发现在数据集Nemotron-Post-Training-Dataset-v1中，每道题被多次采用，且仅保留了正确的COT（Chain of Thought）样本。基于此现象，我们可以计算每道题的准确率和COT长度。整个数据预处理过程分为以下三个步骤：

1. **分离完全正确与部分正确样本**：我们统计了原始数据集中相同题目的COT数量，发现大部分题目对应1-16条COT，极少数题目对应17-32条COT。据此判断，原始数据集对每道题进行了16次推理，然后仅保留了正确的COT样本。其中17-32条的样本可视为少量重复题目。因此，第一步是过滤掉COT数量为16和32的样本（即完全正确的样本），保留部分正确的样本：
    ```bash
    # cd data_preprocess 
    python split_all_right_and_partial_right.py all_samples.jsonl --complete_output all_right_samples.jsonl --incomplete_output partial_right_samples.jsonl --processes 128 
    ```
   原始数据集包含2,044,407条COT数据，经过处理后，我们得到1,189,392条完全正确的COT数据（已过滤掉全对题目）和855,015条部分正确的COT数据。

2. **筛选长COT样本**：从855,015条部分正确的COT数据中筛选出平均COT长度大于32K的样本：
    ```bash
    # cd data_preprocess 
    python static_and_filter_cot.py partial_right_samples.jsonl partial_right_samples_cot_filter.jsonl path_to_tokenizer --processes 128
    ```
    处理后，我们得到约34K条题目，且平均COT长度均超过32K。

3. **提取唯一题目**：从34K条COT数据中提取首次出现的唯一题目：
   
   ```bash
   # cd data_preprocess 
   python extract_first_problem.py partial_right_samples_cot_filter.jsonl partial_right_problem.jsonl
   ```
   最终处理后，我们得到约6K条唯一的题目。


##### 2.2.3 模型采样

获得6K数据集后，我们使用`PCL-Reasoner-V1`模型进行采样，每道题采样8次，生成推理结果。采样配置跟后面的评估一致：

| 采样超参       | 取值                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top_k         | 40                                         |
| top_p         | 0.95                                       |
| max_model_len    | 131072                                     |
| system_prompt_type | amthinking |

经过采样，我们得到了48K的COT数据。

##### 2.2.4 采样COT正确性评估

基于以往训练经验，我们发现传统的math_verify方法无法有效评估COT回答的正确性。对于复杂的数学题，使用规则匹配的方式容易产生较大误判。因此，我们采用Qwen3-32B模型对COT回答的正确性进行评估，具体思路如下：

1. 为`Qwen3-32B`模型设计专门的prompt，用于判断COT中包含的最终答案是否与题目提供的`ground truth`一致；
2. 部署`Qwen3-32B`模型对48K题目进行推理评估；
3. 根据`Qwen3-32B`模型对COT最后300个token中包含的答案与题目提供的`ground truth`进行匹配，从而判断该条COT是否正确。

评估prompt模板如下：

```bash
As a math scoring expert, given a standard answer, and a candidate answer, you need to compare whether the standard answer and the candidate answer are consistent. If they are consistent, return 1; if not, return 0. Remember the returned value should always be put in the \\boxed{}.\nHere are a few points to note:\n1. For the candidate answer, only consider the content inside \\boxed{}, ignoring any other text or error. If no \\boxed{} found, return 0 directly.\n2. If the standard answer and the candidate answer are different but mathematically equivalent, return 1.\n3. For answers involving decimals, if most digits are the same and only the last one or two digits differ, you may considerably return 1.\n4. For all other cases where the standard answer and the candidate answer do not match, return 0.\nHere is a task example:\n<Standard Answer Begin>\n{Standard answer}\n<Standard Answer End>\n<Candidate Answer Begin>\n{Candidate answer}\n<Candidate Answer End>\nPlease put your return value (0 or 1) as required above in the \\boxed{} without any explanation or description.
```

最终，我们获得了22,990条正样本和25,522条负样本。

#### 2.3 模型权重准备

用户可以从`HuggingFace`官方下载`PCL-Reasoner-V1`权重

| 模型名称          | 权重链接                                                                           |
| ----------------- | ---------------------------------------------------------------------------------- |
| PCL-Reasoner-V1 | [https://huggingface.co/PCL-Reasoner/V1](https://huggingface.co/PCL-Reasoner/V1) |

### 3 训练流程

我们的训练基于Mindspeed-LLM框架架，主要包含以下步骤：

#### 3.1 模型权重转换

##### 3.1.1 下载HuggingFace模型权重

下载 HuggingFace 上的 PCL-Reasoner/V1 模型权重到本地。

```bash
# download  model
huggingface-cli download  PCL-Reasoner/V1  --local-dir ~/local/PCL-Reasoner/V1
```

##### 3.1.2 转换模型权重格式

MindSpeed-LLM框架基于MindSpeed，读取权重格式为mcore格式，在训练前，需要将 Hugging Face 权重转换成Mcore格式。脚本启动命令可以用bash启动，可根据真实情况配置脚本，启动命令和配置参数如下：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

hf_model_path=/path/to/hf/model
# 设置需要的权重转换参数
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

###### 参数介绍

- `use-mcore-models`：启用 MCore 模型；
- `model-type`：指定模型类型，如 GPT;
- `load-model-type`：指定加载模型的类型，如 hf（Hugging Face）;
- `save-model-type`：指定保存模型的类型，如 mg;
- `target-tensor-parallel-size`：设置目标张量并行大小；
- `target-pipeline-parallel-size`：设置目标流水线并行大小；
- `add-qkv-bias`：是否进行 QKV 偏置；
- `load-dir`：加载 Hugging Face 权重的路径；
- `save-dir`：保存转换后权重的路径；
- `tokenizer-model`：分词器模型文件的路径；
- `model-type-hf`：指定 Hugging Face 模型类型，如 llama2;
- `params-dtype`：指定参数的数据类型，如 bf16。



#### 3.2 数据集转换

经过推理，我们得到了48K的COT数据，数据格式为jsonl格式，包含问题、推理结果和推理结果对应的COT。需要将其转换为MindSpeed-LLM的可读格式：


```bash
# 请按照您的真实环境修改 set_env.sh 路径
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

#####

- `input`: 输入jsonl文件路径
- `tokenizer-type`: 输入的tokenizer类型
- `tokenizer-name-or-path`: 输入的tokenizer路径
- `output-prefix`: 输出文件路径
- `workers`: 数据处理线程数
- `log-interval`: 日志输出样本数间隔
- `handler-name`: 选择的`data handler`名称
- `prompt-type`: 输入的prompt类型，empty表示无prompt（说明原始prompt里面已经包含了`chat template`)
- `cache-dir`: 缓存目录
- `map-keys`: 输入的jsonl文件字段映射


#### 3.3 训练配置

我们采用了一种受启发的训练策略。全局批大小（global batch size）设为 128，学习率从$6×10^{−5}$按照余弦衰减调度降至 $1×10^{−7}$，并设置了 0.05 的预热比例（warm-up ratio）。AdamW 优化器的动量参数配置为 $\beta_1=0.9$ 和 $\beta_2=0.95$。训练在 16 台 Atlas 910C SuperPoD 节点（每台包含 8 个芯片）上进行。整个微调过程共进行了 4 个 epoch，耗时约 116 小时。相应的训练损失曲线如图 \ref{fig:loss} 所示。

为了最大化计算效率，我们在监督微调阶段启用了数据打包（data packing）功能。该功能允许将每个批次中长度各异的样本拼接合并，填入预设的序列长度（48K tokens）中。通过将多个短序列合并为一个长序列，有效消除了因序列填充（padding）带来的冗余计算，显著加快了训练速度。

#### 3.4 启动训练

训练流程分为三步：

1. 激活环境：`source /path/to/set_env.sh`
2. 启动训练：`cd MindSpeed-LLM; bash scripts/lauch_multi_nodes.sh node_list.txt`

### 4. 评测流程：

我们使用 [LLMEval](https://gitee.com/jianzhnie/LLMEval) 对模型进行评测， LLMEval 是由我们团队开发的主要针对大模型推理进行评测的工具，支持 vllm 和 sglang 两种推理后端， 支持多种评测数据集， 已经在 Ascend 环境复现了多个开源推理模型的效果。详情请参考 [LLMEval 使用教程](https://gitee.com/jianzhnie/LLMEval)。

#### 4.1 评估环境配置

请参考 [LLMEval 使用教程](https://gitee.com/jianzhnie/LLMEval) 中的环境配置章节进行环境搭建。


#### 4.2 开始评测

##### 步骤 1：启动 vLLM 服务器

```bash
source set_env.sh

model_path="/path/to/pcl_reasoner_v1"
model_name="PCL-Reasoner-v1"

num_gpus=8
max_model_len=131072  # ✅ 支持 128k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

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

根据可用设备调整 `tensor_parallel_size` 参数。


##### 步骤 2：提交推理任务

启动 vLLM 服务后，运行推理脚本生成响应, 并将结果保存到指定的输出文件中。

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

**注意：** 我们使用重复采样来减少评估方差，但可能需要较长时间才能完成（根据设备情况可能超过8小时）。


我们采用的评测超参如下所示：

| 采样超参       | 取值                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top_k         | 40                                         |
| top_p         | 0.95                                       |
| max_model_len    | 131072                                     |
| system_prompt_type | amthinking |

##### 步骤 3：评分

完成推理后，使用以下命令进行评分：

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


####  4.3 评测结果

我们在AIME24/AIME25评测结果详见下表数据。为确保评估准确性，我们采用Avg@32指标（平均32次采样）进行了评测：

<!-- 表格基础样式（可选添加） -->

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

> *(注：模型在AIME24/25评测集上的生成结果文件已同步上传至 `pcl_reasoner_v1/eval/eval_res`目录，供开发者用于模型验证与效果比对参考）*
 

另外，我们也针对评测时不同模型回答长度统计正确率，可以看出AIME24/25评测集对回答长度要求较高，而且较为简单的AIME24上，64K tokens的回答长度可以满足，而较为难的AIME25上则需要回答长度长达128K tokens：

<style>
  table { border-collapse: collapse; width: 100%; margin-left: auto;margin-right: auto;}
  th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
</style>

<table>
  <tr>
    <th>回答长度</th>
    <th>16k</th>
    <th>32k</th>
    <th>64k</th>
    <th>128k</th>
  </tr>
  <tr>
    <td>AIME24</td>
    <td>42.0</td>
    <td>77.9</td>
    <td>85.7</td>
    <td>85.7</td>
  </tr>
  <tr>
    <td>AIME25</td>
    <td>33.4</td>
    <td>75.6</td>
    <td>83.9</td>
    <td>84.2</td>
  </tr>
</table>

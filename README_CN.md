# **PCL-Reasoner-V1.5 数学推理模型**

## 模型概览

本次发布的PCL-Reasoner-V1.5模型，以PCL-Reasoner-V1为起点，基于MindSpeed-LLM框架与Ascend硬件进行了高性能的监督微调。经过微调，模型在数学推理能力上取得了显著提升：其在权威基准评测集AIME24上准确率达 9x.x%，AIME25上达 9x.x%，在 32B参数级别模型中稳居前列。
为促进技术共享与应用，我们已完整开源了PCL-Reasoner-V1.5的模型权重、微调数据及训练代码。该模型不仅是当下领先的32B数学推理模型之一，更为开发者提供了宝贵的专业领域监督微调实践经验与后训练解决方案。用户可参照以下教程轻松部署体验，深入探索后训练的实践方法与奥秘！


## 开发指导

### 1. 模型文件

PCL-Reasoner-V1.5基于PCL-Reasoner-V1进行微调后训练，训练流程基于MindSpeed-LLM框架实现，主要涉及的文件有：


### 2.环境及数据准备

#### 2.1 安装环境：

| 软件      | 版本       |
| --------- | ---------- |
| 固件&驱动 | 24.1.rc3.5 |
| CANN      | 8.2.RC1    |
| Python    | 3.10       |


#### 2.2 数据处理

##### 2.2.1 数据集下载

要想进一步提升PCL-Reasoner-V1的能力，我们考虑从Nvidia公开的`Nemotron-Post-Training-Dataset-v1`中寻找具备一定难度的题目来做进一步的训练。

| 数据集名称                    | 数据集链接                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| nvidia/Nemotron-Post-Training-Dataset-v1 | [https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) |

##### 2.2.2 数据预处理

数据集下载后为parquet格式，我们首先将数据集转换为jsonl格式，方便后续处理。

```bash
# dir_path_to_parquet_files=Nemotron-Post-Training-Dataset-v1/data
# output_dir_path=Nemotron-Post-Training-Dataset-v1/orig2jsonl
cd PCL-Reasoner-V1.5/data_preprocess
python convert_parquet2jsonl.py $dir_path_to_parquet_files $output_dir_path  --workers 128
# 将数据集合并为一个jsonl文件
cat $output_dir_path/*jsonl > Nemotron-Post-Training-Dataset-v1/all_samples.jsonl
```

经过统计，我们发现在数据集`Nemotron-Post-Training-Dataset-v1`中，每道题被采用了多次，并且只保留了正确的COT样本。 因此我们可以据此计算每道题的准确率和COT长度。我们的数据处理分为了3步：

1. 我们统计了原始数据集相同题目的COT个数，发现大部分都处于1-16条，还有极少数的处于17-32条的区间。因此，我们判断原始数据集是对每个题目推理了16次，然后只保留了正确的COT样本。其中17-32条的样本可以理解为有少数的题目重复了。因此我们第一步就是去掉COT条数为16和32的样本，即全对的样本，保留推理部分正确的样本：
   ```bash
   cd PCL-Reasoner-V1.5/data_preprocess 
   python split_all_right_and_partial_right.py all_samples.jsonl --complete_output all_right_samples.jsonl --incomplete_output partial_right_samples.jsonl --processes 128 
   ```
   原始数据集有2044407条COT数据，经过处理，我们得到了1189392条完全正确的COT数据（全对的题目过滤掉），和855015条部分正确的COT数据。

2. 接下来，我们从855015条部分正确的COT数据中筛选出平均COT长度大于32K的COT数据：
    ```bash
    cd PCL-Reasoner-V1.5/data_preprocess 
    python static_and_filter_cot.py partial_right_samples.jsonl partial_right_samples_cot_filter.jsonl path_to_tokenizer --processes 128
    ```
    经过处理，我们只得到了34K的题目，且平均COT长度大于32K。

3. 最后我们再从34K的COT中，找出唯一出现的题目:
   
   ```bash
   cd PCL-Reasoner-V1.5/data_preprocess 
   python extract_first_problem.py partial_right_samples_cot_filter.jsonl partial_right_problem.jsonl
   ```
   经过处理，我们最终得到了6K的题目。


##### 2.2.3 模型采样

我们得到这6K数据集后，利用`PCL-Reasoner-V1`模型进行采样，每道题采样8次，生成推理结果。采样的配置如下：

xxx

经过采样，我们得到了48K的COT数据。

#### 2.2.4 采样COT正确性评估

我们在以往的训练经验中发现，采用`math_verify`并不能很好的对COT回答的正确性进行评估。对于越是复杂的数学题，其它答案如果采用规则进行匹配，那么就会有较大的误判。因此，我们采用`Qwen3-32B`模型来对COT的回答正确性进行评估。具体思路如下：

1. 为`Qwen3-32B`模型专门写一个prompt，用于判断COT的最后里面包含的答案是否与题目提供的ground truth一致；
2. 部署`Qwen3-32B`模型对48K题目进行推理；
3. 根据`Qwen3-32B`模型对COT的最后300个token里面包含的答案和题目提供的ground truth进行匹配从然判断该条COT是否正确。

prompt模板如下：

```bash
As a math scoring expert, given a standard answer, and a candidate answer, you need to compare whether the standard answer and the candidate answer are consistent. If they are consistent, return 1; if not, return 0. Remember the returned value should always be put in the \\boxed{}.\nHere are a few points to note:\n1. For the candidate answer, only consider the content inside \\boxed{}, ignoring any other text or error. If no \\boxed{} found, return 0 directly.\n2. If the standard answer and the candidate answer are different but mathematically equivalent, return 1.\n3. For answers involving decimals, if most digits are the same and only the last one or two digits differ, you may considerably return 1.\n4. For all other cases where the standard answer and the candidate answer do not match, return 0.\nHere is a task example:\n<Standard Answer Begin>\n{Standard answer}\n<Standard Answer End>\n<Candidate Answer Begin>\n{Candidate answer}\n<Candidate Answer End>\nPlease put your return value (0 or 1) as required above in the \\boxed{} without any explanation or description.\n<|im_end|>
```

最终，我们得到了22990条正样本和25522条负样本。

#### 2.3 模型权重准备

用户可以从`HuggingFace`官方下载`PCL-Reasoner-V1`权重

| 模型名称          | 权重链接                                                                           |
| ----------------- | ---------------------------------------------------------------------------------- |
| PCL-Reasoner-V1 | [https://huggingface.co/PCL-Reasoner/V1](https://huggingface.co/PCL-Reasoner/V1) |

### 3 训练流程

我们的训练基于Mindspeed-LLM框架架，主要包含以下步骤：

#### 3.1 模型权重转换

MindSpeed-LLM框架基于MindSpeed，读取权重格式为mcore格式，因此需要转换权重格式。


#### 3.2 数据集转换

经过推理，我们得到了48K的COT数据，数据格式为jsonl格式，包含问题、推理结果和推理结果对应的COT。需要将其转换为MindSpeed-LLM的可读格式：


#### 3.3 训练配置


#### 3.4 启动训练

### 4. 评测流程：


我们采用的评测超参如下所示：

| 采样超参       | 取值                                       |
| -------------- | ------------------------------------------ |
| temperature    | 0.6                                        |
| top\_k         | 40                                         |
| top\_p         | 0.95                                       |
| max\_tokens    | 129024                                     |
| chat\_template | `./pcl_reasoner_v1/eval/am_thinking.jinja` |

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


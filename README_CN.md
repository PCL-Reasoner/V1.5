# **PCL-Reasoner-V1.5 数学推理模型**

## 模型概览

本次发布的PCL-Reasoner-V1.5模型，以PCL-Reasoner-V1为起点，基于MindSpeed-LLM框架与Ascend硬件进行了高性能的监督微调。经过微调，模型在数学推理能力上取得了显著提升：其在权威基准评测集AIME24上准确率达 85.7%，AIME25上达 84.2%，在 32B参数级别模型中稳居前列。
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

用户可以从HuggingFace官方下载原始数据集：

| 数据集名称                    | 数据集链接                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| AM-DeepSeek-R1-0528-Distilled | [https://huggingface.co/a-m-team/AM-DeepSeek-R1-0528-Distilled](https://huggingface.co/a-m-team/AM-DeepSeek-R1-0528-Distilled) |

##### 2.2.2 数据预处理

首先，我们对源数据进行检测和筛选，操作分为两个步骤，验证集污染检测与数据筛选。

* 验证集污染检测：我们采用基于all-MiniLM-L6-v2模型计算文本余弦相似度的方法，对数学部分原始数据针对AIME24/25评测集进行污染检测。该脚本执行后会在终端打印检测结果，并在指定的输出路径中保存相似度大于阈值的题目及其匹配的评测集题目。
  
  ```
  python PCL-Reasoner-V1/pcl_reasoner_v1/data_preprocess/decontaminate.py \
  --target_data /path/to/target_data \
  --contaminant_source PCL-Reasoner-V1/pcl_reasoner_v1/data_preprocess/aime2425_questions.json \
  --model_path /path/to/distilled/model_path \
  --output_file_prefix /path/to/output_file_prefix
  --threshold 0.7
  
  # 参数说明
  target_data：需要被检测的数据
  contaminant_source：污染源，即评测集数据
  model_path：计算文本嵌入的模型
  output_file_prefix：检测结果输出的路径
  threshold：相似度阈值
  ```

### 3 训练流程
#### 3.1 权重准备

用户可以从HuggingFace官方下载预训练权重

| 模型名称          | 权重链接                                                                           |
| ----------------- | ---------------------------------------------------------------------------------- |
| qwen2\_5-32b-Base | [https://huggingface.co/Qwen/Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) |

MindFormers 1.5.0及以上版本已支持safetensor格式的权重直接加载及保存，无需转换成ckpt，下文中微调将使用safetensors格式权重运行。

#### 3.2 训练配置：

下面仅列出用户高频修改的配置项，完整配置文件见`pcl_reasoner_v1/config/ finetune_pcl_reasoner_v1_32k.yaml`


#### 3.3 启动微调

在启动脚本`run_pcl_reasoner_v1_finetune.sh`指定配置文件`pcl_reasoner _v1/config/finetune_pcl_reasoner_v1_32k.yaml`，并根据用户的实际情况对卡数、服务器IP等配置进行修改：


### 4. 评测流程：

为了保障评测结果的公平性，我们采用了QwQ开源的评测代码（QwQ/eval at main · QwenLM/QwQ），可以根据代码仓中README.md指导进行环境安装及模型评测。
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


# ​**PCL-Reasoner-V1.5**​

## Model Overview  
We release **PCL-Reasoner-V1.5**, a next-generation reasoning model built upon **PCL-Reasoner-V1** and further enhanced through **offline reinforcement learning** method on the **MindSpeed-LLM framework** with **Ascend hardware acceleration**. Building on the strong foundation of PCL-Reasoner-V1, PCL-Reasoner-V1.5 achieves even greater proficiency in complex mathematical reasoning, demonstrating state-of-the-art performance among 32B-scale models.

PCL-Reasoner-V1.5 attains **9x.x% on AIME 2024** and **9x.x% on AIME 2025**, significantly outperforming prior 32B-class models and closing the gap with much larger systems. This advancement stems from **~~refined data curation, improved contamination filtering, and optimized training dynamics~~** tailored for deep reasoning tasks.

![Evaluation Results](images/benchmark.png)

We have fully open-sourced the **model weights**, **dataset**, and **training code** to foster transparency, reproducibility, and community innovation. Follow the tutorial below to deploy, evaluate, or extend PCL-Reasoner-V1.5 in your own research!

## Code  
[GitHub Repository](https://github.com/your-org/XXX)
[OpenI Project Page](https://openi.pcl.ac.cn/your-org/XXX)

## Evaluation  
All results are reported using the **Avg@32 metric** (average accuracy over 32 independent sampling attempts per problem), ensuring robust and fair comparison.

| Parameter Size | Model Name                          | AIME 24 | AIME 25 |
|----------------|-------------------------------------|---------|---------|
| >100B          | DeepSeek-R1                         | <span style="color:grey">79.8</span> | <span style="color:grey">70.0</span> |
|                | DeepSeek-R1-0528                    | <span style="color:grey">91.4</span> | <span style="color:grey">87.5</span> |
|                | DeepSeek-V3.2-Speciale              | <span style="color:grey"></span> | <span style="color:grey">96.0</span> |
|                | DeepSeek-V3.2-Thinking              | <span style="color:grey"></span> | <span style="color:grey">93.1</span> |
|                | GPT-5-High                          | <span style="color:grey"></span> | <span style="color:grey">94.6</span> |
|                | Claude-4.5-Sonnet                   | <span style="color:grey"></span> | <span style="color:grey">87.0</span> |
|                | Qwen3-235B-A22B                     | <span style="color:grey">85.7</span> | <span style="color:grey">81.5</span> |
|                | OpenAI-o3                           | <span style="color:grey">91.6</span> | <span style="color:grey">88.9</span> |
|                | Gemini-2.5-Pro-0506                 | <span style="color:grey">90.8</span> | <span style="color:grey">83.0</span> |
|                | Gemini-3-Pro                      | <span style="color:grey"></span> | <span style="color:grey">95.0</span> |
|                | Qwen3-Max-Instruct                  | <span style="color:grey"></span> | <span style="color:grey">81.6</span> |
|                | Qwen3-Max-Thinking                  | <span style="color:grey"></span> | <span style="color:grey">100.0</span> |
| 32B            | Qwen3-32B                           | <span style="color:grey">81.4</span> | <span style="color:grey">72.9</span> |
|                | QwQ-32B                             | <span style="color:grey">79.5</span> | <span style="color:grey">69.5</span> |
|                | DeepSeek-R1-Distill-Qwen-32B        | <span style="color:grey">72.6</span> | <span style="color:grey">49.6</span> |
|                | Skywork-OR1-32B                     | <span style="color:grey">82.2</span> | <span style="color:grey">73.3</span> |
|                | AM-Thinking-v1                      | <span style="color:grey">85.3</span> | <span style="color:grey">74.4</span> |
|                | PCL-Reasoner-V1                     | **85.7** | **84.2** |
|                | OpenReasoning-Nemotron-32B          | <span style="color:grey">89.2</span> | <span style="color:grey">84.0</span> |
|                | **PCL-Reasoner-V1.5**                             | **9x.x** | **9x.x** |

> **Note**: Model outputs on AIME24/25 are included in the repository under `eval/` for verification and analysis.

> **Refenrences**:
> 1. OpenReasoning-Nemotron:  https://huggingface.co/nvidia/OpenReasoning-Nemotron-14B
> 2. Qwen3-Max: https://qwen.ai/blog?id=qwen3-max
> 3. DeepSeek-V3.2-Speciale: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
> 4. AIME25 leaderboard: https://artificialanalysis.ai/evaluations/aime-2025
> 5. AIME24 leaderboard: https://llm-stats.com/benchmarks/aime-2024

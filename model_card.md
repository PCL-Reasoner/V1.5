# **PCL-Reasoner-V1.5**

## Model Overview  
We release **PCL-Reasoner-V1.5**, a next-generation reasoning model built upon **PCL-Reasoner-V1** and further enhanced through **offline reinforcement learning** method on the **vllm-ascend** and **MindSpeed-LLM framework** with **Ascend hardware acceleration**. Building on the strong foundation of PCL-Reasoner-V1, PCL-Reasoner-V1.5 achieves even greater improvement in complex mathematical reasoning with long chains of thought (CoT), demonstrating state-of-the-art performance among 32B-scale models.

PCL-Reasoner-V1.5 attains **90.9% on AIME 2024** and **85.7% on AIME 2025**, significantly outperforming prior 32B-class models and closing the gap with much larger systems. This advancement stems from refined data curation, improved contamination filtering, and optimized training dynamics tailored for deep reasoning tasks.

![Evaluation Results](images/benchmark.png)

We have fully open-sourced the **model weights**, **dataset**, and **training code** to foster transparency, reproducibility, and community innovation. Follow the tutorial below to deploy, evaluate, or extend PCL-Reasoner-V1.5 in your own research!

## Codes

[GitHub Repository](https://github.com/PCL-Reasoner/V1.5)

[OpenI Project Page](https://openi.pcl.ac.cn/PCL-Reasoner/V1.5)

## Evaluation  
All results are reported using the **Avg@32 metric** (average accuracy over 32 independent sampling attempts per problem), ensuring robust and fair comparison.

<!-- Table base styling (optional) -->

<style>
  table { border-collapse: collapse; width: 100%; margin-left: auto;margin-right: auto;}
  th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
</style>

<!-- Table content -->

<table>
  <tr>
    <th>Model Scale</th>
    <th>Model</th>
    <th>AIME 24</th>
    <th>AIME 25</th>
  </tr>
  <!-- Merged row header >100B -->
  <tr>
    <th rowspan="6">&gt;100B</th>
  </tr>
  <!-- >100B data rows -->
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
  <!-- Separator row -->
  <tr>
    <td colspan="4"></td>
  </tr>
  <!-- Merged row header 32B -->
  <tr>
    <th rowspan="9">32B</th>
  </tr>
  <!-- 32B data rows -->
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
    <td>OpenReasoning-Nemotron-32B</td>
    <td><span style="color:grey">89.2</span></td>
    <td><span style="color:grey">84.2</span></td>
  </tr>
  <tr>
    <td>PCL-Reasoner-v1</td>
    <td><p style="font-weight:grey;">85.7</p></td> 
    <td><p style="font-weight:grey;">84.2</p></td> 
  </tr>
  <tr>
    <td>PCL-Reasoner-v1.5</td>
    <td><p style="font-weight: bold;">90.9</p></td> 
    <td><p style="font-weight: bold;">85.7</p></td> 
  </tr>
</table>

> **Note**: Model outputs on AIME24/25 are included in the repository under `eval_result/` for verification and analysis.

## Citation

```bibtex
@article{PCL-Reasoner-v1.5,
  title={PCL-Reasoner-v1.5: A Math Problem Solver with Chain of Thought Reasoning},
  author={Yao Lu, Deng Dong Fan, Jianzheng Nie, et al.},
  journal={arXiv preprint arXiv:2405.14524},
  year={2026}
}
```

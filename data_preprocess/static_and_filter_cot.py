import json
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import torch
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str):
    """
    加载tokenizer
    :param tokenizer_path: tokenizer路径
    :return: tokenizer对象
    """
    return AutoTokenizer.from_pretrained(tokenizer_path)


def process_samples_chunk(args):
    """
    处理样本块，计算COT长度并添加统计信息
    :param args: 包含(chunk_data, tokenizer_path)的元组
    :return: 处理后的样本列表
    """
    chunk_data, tokenizer_path = args
    tokenizer = load_tokenizer(tokenizer_path)
    processed_samples = []
    
    for sample in chunk_data:
        problem = sample['problem']
        cot = sample.get('cot', '')
        
        # 计算COT长度
        cot_length = len(tokenizer.encode(cot))
        
        # 创建新样本，添加统计信息
        new_sample = sample.copy()
        new_sample['cot_length'] = cot_length
        
        processed_samples.append((problem, new_sample))
    
    return processed_samples


def aggregate_problem_stats(processed_chunks: List[List[Tuple[str, dict]]]) -> Dict[str, List[dict]]:
    """
    汇总每个problem的样本
    :param processed_chunks: 处理后的样本块列表
    :return: 按problem分组的样本字典
    """
    problem_samples = defaultdict(list)
    
    for chunk in processed_chunks:
        for problem, sample in chunk:
            problem_samples[problem].append(sample)
    
    return problem_samples


def calculate_stats_and_update_samples_with_filtering(
    problem_samples: Dict[str, List[dict]], 
    min_avg_length: int = 32000
) -> tuple:
    """
    计算每个problem的统计信息并更新样本，过滤平均长度小于阈值的样本
    :param problem_samples: 按problem分组的样本字典
    :param min_avg_length: 最小平均COT长度阈值
    :return: (更新后的样本列表, COT长度区间统计, 出现次数区间统计)
    """
    updated_samples = []
    cot_length_counts = defaultdict(int)  # COT长度区间统计
    occurrence_counts = defaultdict(int)  # 出现次数区间统计
    
    for problem, samples in problem_samples.items():
        # 计算平均COT长度
        total_cot_length = sum(sample['cot_length'] for sample in samples)
        avg_cot_length = total_cot_length / len(samples) if samples else 0
        
        # 只有当平均COT长度大于等于阈值时才保留样本
        if avg_cot_length >= min_avg_length:
            # 计算COT长度区间（每1000为一个区间）
            for sample in samples:
                cot_len = sample['cot_length']
                cot_length_interval = (cot_len // 1000) * 1000
                cot_length_counts[cot_length_interval] += 1
            
            # 计算出现次数区间（每2个为一个区间）
            occurrence_interval = (len(samples) // 2) * 2
            occurrence_counts[occurrence_interval] += 1
            
            # 更新每个样本，添加统计信息
            for sample in samples:
                sample['count'] = len(samples)  # 添加样本个数
                sample['avg_cot_length'] = avg_cot_length  # 添加平均COT长度
                del sample['cot_length']  # 删除临时字段
                updated_samples.append(sample)
    
    return updated_samples, cot_length_counts, occurrence_counts


def print_statistics(cot_length_counts: Dict[int, int], occurrence_counts: Dict[int, int]):
    """
    打印统计信息
    :param cot_length_counts: COT长度区间统计
    :param occurrence_counts: 出现次数区间统计
    """
    print("\n=== COT长度区间统计 ===")
    sorted_cot_intervals = sorted(cot_length_counts.keys())
    for interval in sorted_cot_intervals:
        lower_bound = interval
        upper_bound = interval + 1000
        count = cot_length_counts[interval]
        print(f"COT长度 [{lower_bound}, {upper_bound}): {count} 个样本")
    
    print("\n=== 出现次数区间统计 ===")
    sorted_occurrence_intervals = sorted(occurrence_counts.keys())
    for interval in sorted_occurrence_intervals:
        lower_bound = interval
        upper_bound = interval + 2
        count = occurrence_counts[interval]
        print(f"出现次数 [{lower_bound}, {upper_bound}): {count} 个问题")


def read_jsonl(file_path: str, chunk_size: int = 10000) -> List[List[dict]]:
    """
    将JSONL文件分割成多个块
    :param file_path: JSONL文件路径
    :param chunk_size: 每个块的大小
    :return: 分割后的数据块列表
    """
    chunks = []
    current_chunk = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    current_chunk.append(sample)
                    
                    if len(current_chunk) >= chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = []
                except json.JSONDecodeError:
                    print(f"警告: 第{line_num+1}行JSON格式错误，跳过该行")
                    continue
    
    # 添加最后一个不完整的块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def write_jsonl(data: List[dict], output_path: str):
    """
    将数据写入JSONL文件
    :param data: 要写入的数据列表
    :param output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main(input_file: str, output_file: str, tokenizer_path: str, 
         min_avg_length: int = 32000, num_processes: int = None, chunk_size: int = 10000):
    """
    主函数：统计COT长度并过滤样本，同时进行区间分析
    :param input_file: 输入JSONL文件路径
    :param output_file: 输出JSONL文件路径
    :param tokenizer_path: tokenizer路径
    :param min_avg_length: 最小平均COT长度阈值（默认32000）
    :param num_processes: 进程数，默认为CPU核心数
    :param chunk_size: 每个处理块的大小
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"开始处理文件: {input_file}")
    print(f"使用tokenizer: {tokenizer_path}")
    print(f"使用进程数: {num_processes}")
    print(f"分块大小: {chunk_size}")
    print(f"最小平均COT长度阈值: {min_avg_length}")
    
    # 分割文件为多个块
    print("正在分割文件...")
    chunks = read_jsonl(input_file, chunk_size)
    print(f"文件已分割为 {len(chunks)} 个块")
    
    # 准备多进程参数
    process_args = [(chunk, tokenizer_path) for chunk in chunks]
    
    # 使用多进程处理每个块
    print("开始多进程处理...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_chunks = pool.map(process_samples_chunk, process_args)
    
    # 汇总每个problem的样本
    print("汇总每个问题的样本...")
    problem_samples = aggregate_problem_stats(processed_chunks)
    
    # 计算统计信息并更新样本，过滤低平均长度的样本
    print("计算统计信息并过滤样本...")
    updated_samples, cot_length_counts, occurrence_counts = calculate_stats_and_update_samples_with_filtering(
        problem_samples, min_avg_length
    )
    
    # 打印统计信息
    print_statistics(cot_length_counts, occurrence_counts)
    
    # 写入输出文件
    print("写入处理后的文件...")
    write_jsonl(updated_samples, output_file)
    
    # 输出最终统计信息
    print(f"\n处理完成!")
    print(f"过滤后样本总数: {len(updated_samples)}")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计样本COT长度并进行区间分析，过滤平均长度小于阈值的样本")
    parser.add_argument("input_file", help="输入JSONL文件路径")
    parser.add_argument("output_file", help="输出JSONL文件路径")
    parser.add_argument("tokenizer_path", help="tokenizer路径")
    parser.add_argument("--min_avg_length", type=int, default=32000, 
                        help="最小平均COT长度阈值 (默认: 32000)")
    parser.add_argument("--processes", type=int, default=None, 
                        help="进程数 (默认: CPU核心数)")
    parser.add_argument("--chunk_size", type=int, default=10000, 
                        help="每个处理块的大小 (默认: 10000)")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        tokenizer_path=args.tokenizer_path,
        min_avg_length=args.min_avg_length,
        num_processes=args.processes,
        chunk_size=args.chunk_size
    )

    """
    python static_and_filter_cot.py partial_right_samples.jsonl partial_right_samples_cot_filter.jsonl /home/fdd/workspace/models/Qwen/Qwen2.5-32B --processes 128
    """


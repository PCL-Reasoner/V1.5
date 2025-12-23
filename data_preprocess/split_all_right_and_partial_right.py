"""
为数据集分离为全对（出现16次，32次等）和非全对的样本
"""
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import argparse


def process_chunk(chunk_data: List[Tuple[int, dict]]) -> dict:
    """
    处理数据块，统计每个problem的出现次数及对应的样本
    :param chunk_data: 包含(行号, 样本数据)的列表
    :return: 包含统计信息的字典
    """
    problem_samples = defaultdict(list)
    
    for line_num, sample in chunk_data:
        problem = sample.get('problem', '')
        problem_samples[problem].append((line_num, sample))
    
    return problem_samples


def read_jsonl_chunks(file_path: str, chunk_size: int = 10000) -> List[List[Tuple[int, dict]]]:
    """
    将JSONL文件分割成多个块
    :param file_path: JSONL文件路径
    :param chunk_size: 每个块的行数
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
                    current_chunk.append((line_num, sample))
                    
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


def merge_results(results: List[dict]) -> dict:
    """
    合并多进程处理的结果
    :param results: 多进程返回的结果列表
    :return: 合并后的统计字典
    """
    merged = defaultdict(list)
    
    for result in results:
        for problem, samples in result.items():
            merged[problem].extend(samples)
    
    return merged


def write_jsonl(data: List[dict], output_path: str):
    """
    将数据写入JSONL文件
    :param data: 要写入的数据列表
    :param output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main(input_file: str, complete_output: str, incomplete_output: str, num_processes: int = None, chunk_size: int = 10000):
    """
    主函数：分离JSONL文件中的完整和不完整样本
    :param input_file: 输入JSONL文件路径
    :param complete_output: 完整样本输出文件路径
    :param incomplete_output: 不完整样本输出文件路径
    :param num_processes: 进程数，默认为CPU核心数
    :param chunk_size: 每个处理块的大小
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"开始处理文件: {input_file}")
    print(f"使用进程数: {num_processes}")
    print(f"分块大小: {chunk_size}")
    
    # 分割文件为多个块
    print("正在分割文件...")
    chunks = read_jsonl_chunks(input_file, chunk_size)
    print(f"文件已分割为 {len(chunks)} 个块")
    
    # 使用多进程处理每个块
    print("开始多进程处理...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # 合并所有结果
    print("合并处理结果...")
    all_samples = merge_results(results)
    
    # 分离完整和不完整样本
    complete_samples = []
    incomplete_samples = []
    
    complete_counts = [16, 32, 48]  # 定义完整样本的计数
    
    total_problems = len(all_samples)
    processed_count = 0
    
    for problem, samples in all_samples.items():
        count = len(samples)
        
        if count%16 == 0:
            complete_samples.extend([s[1] for s in samples])  # 只保存样本数据，不保存行号
        else:
            incomplete_samples.extend([s[1] for s in samples])
        
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"已处理 {processed_count}/{total_problems} 个问题")
    
    # 写入输出文件
    print("写入完整样本文件...")
    write_jsonl(complete_samples, complete_output)
    
    print("写入不完整样本文件...")
    write_jsonl(incomplete_samples, incomplete_output)
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"完整样本数量: {len(complete_samples)} (问题数: {len([p for p, s in all_samples.items() if len(s) in complete_counts])})")
    print(f"不完整样本数量: {len(incomplete_samples)} (问题数: {len([p for p, s in all_samples.items() if len(s) not in complete_counts])})")
    print(f"完整样本输出文件: {complete_output}")
    print(f"不完整样本输出文件: {incomplete_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分离JSONL文件中的完整和不完整样本")
    parser.add_argument("input_file", help="输入JSONL文件路径")
    parser.add_argument("--complete_output", default="complete_samples.jsonl", 
                        help="完整样本输出文件路径 (默认: complete_samples.jsonl)")
    parser.add_argument("--incomplete_output", default="incomplete_samples.jsonl", 
                        help="不完整样本输出文件路径 (默认: incomplete_samples.jsonl)")
    parser.add_argument("--processes", type=int, default=None, 
                        help="进程数 (默认: CPU核心数)")
    parser.add_argument("--chunk_size", type=int, default=16000, 
                        help="每个处理块的大小 (默认: 10000)")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input_file,
        complete_output=args.complete_output,
        incomplete_output=args.incomplete_output,
        num_processes=args.processes,
        chunk_size=args.chunk_size
    )

"""
python split_all_right_and_partial_right.py all_samples.jsonl --complete_output all_right_samples.jsonl --incomplete_output partial_right_samples.jsonl --processes 128 
"""

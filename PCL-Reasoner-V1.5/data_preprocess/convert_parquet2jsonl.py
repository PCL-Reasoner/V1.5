#!/usr/bin/env python3
"""
多进程Parquet转JSONL转换器
支持批量处理目录中的所有Parquet文件，每个文件转换为对应的JSONL文件
"""

import pandas as pd
import json
import multiprocessing as mp
from pathlib import Path
import time
import argparse
import logging
from typing import List, Dict, Any
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_single_file(file_info: tuple) -> Dict[str, Any]:
    """
    处理单个Parquet文件并转换为JSONL格式
    
    Args:
        file_info: 元组包含 (输入文件路径, 输出目录路径)
    
    Returns:
        处理结果字典
    """
    input_file_path, output_dir = file_info
    start_time = time.time()
    
    try:
        input_path = Path(input_file_path)
        output_path = Path(output_dir) / f"{input_path.stem}.jsonl"
        
        logger.info(f"开始处理: {input_path.name}")
        
        # 读取Parquet文件
        df = pd.read_parquet(input_path)
        total_rows = len(df)
        
        # 转换为JSONL并写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                # 将行数据转换为字典
                row_dict = row.to_dict()
                
                # 转换每个样本的数据
                cleaned_dict = row_dict
                cleaned_dict["problem"] = row_dict['messages'][0]['content']
                cleaned_dict["cot"] = row_dict['messages'][1]['content']
                cleaned_dict['ground_truth'] = json.loads(row_dict['metadata'])["expected_answer"]
                cleaned_dict['problem_source'] = json.loads(row_dict['metadata'])["problem_source"]
                cleaned_dict.pop('messages')
                cleaned_dict.pop('metadata')
                
                # 转换为JSON并写入
                json_line = json.dumps(cleaned_dict, ensure_ascii=False)
                f.write(json_line + '\n')
                
                # 每处理10000行打印进度
                if (idx + 1) % 10000 == 0:
                    logger.info(f"{input_path.name}: 已处理 {idx + 1}/{total_rows} 行")
        
        processing_time = time.time() - start_time
        logger.info(f"完成: {input_path.name} -> {total_rows}行, 耗时: {processing_time:.2f}s")
        
        return {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'rows_processed': total_rows,
            'success': True,
            'processing_time': processing_time,
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"处理文件 {input_file_path} 时出错: {str(e)}"
        logger.error(error_msg)
        
        # 清理可能生成的不完整文件
        if 'output_path' in locals() and output_path.exists():
            try:
                output_path.unlink()
                logger.info(f"已删除不完整文件: {output_path}")
            except:
                pass
        
        return {
            'input_file': str(input_file_path),
            'output_file': None,
            'rows_processed': 0,
            'success': False,
            'processing_time': processing_time,
            'error': str(e)
        }

def find_parquet_files(input_dir: str) -> List[Path]:
    """
    查找目录中的所有Parquet文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        Parquet文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有parquet文件（包括子目录）
    parquet_files = list(input_path.rglob("*.parquet"))
    
    # 如果没有找到文件，尝试其他常见扩展名
    if not parquet_files:
        parquet_files = list(input_path.rglob("*.parq"))
    
    return parquet_files

def process_files_parallel(
    input_dir: str, 
    output_dir: str, 
    max_workers: int = None,
    chunk_size: int = 10000
) -> List[Dict[str, Any]]:
    """
    使用多进程并行处理所有Parquet文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        max_workers: 最大工作进程数
        chunk_size: 处理大文件时的分块大小
        
    Returns:
        所有文件处理结果的列表
    """
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有Parquet文件
    parquet_files = find_parquet_files(input_dir)
    if not parquet_files:
        logger.warning(f"在目录 {input_dir} 中未找到Parquet文件")
        return []
    
    logger.info(f"找到 {len(parquet_files)} 个Parquet文件")
    
    # 准备文件处理信息
    file_infos = [(str(file), output_dir) for file in parquet_files]
    
    # 设置进程数
    if max_workers is None:
        # 默认使用CPU核心数，但不超过文件数量
        max_workers = min(mp.cpu_count(), len(parquet_files))
    
    logger.info(f"使用 {max_workers} 个进程并行处理")
    
    # 使用进程池处理文件
    results = []
    with mp.Pool(processes=max_workers) as pool:
        # 使用imap_unordered获取实时进度
        for i, result in enumerate(pool.imap_unordered(process_single_file, file_infos)):
            results.append(result)
            logger.info(f"进度: {i + 1}/{len(parquet_files)} 文件完成")
    
    return results

def generate_summary_report(results: List[Dict[str, Any]]) -> None:
    """
    生成处理结果摘要报告
    
    Args:
        results: 处理结果列表
    """
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    total_files = len(results)
    total_rows = sum(r.get('rows_processed', 0) for r in successful)
    total_time = sum(r.get('processing_time', 0) for r in results)
    avg_time_per_file = total_time / total_files if total_files > 0 else 0
    
    print("\n" + "="*60)
    print("📊 PARQUET转JSONL处理结果摘要")
    print("="*60)
    print(f"总文件数: {total_files}")
    print(f"成功处理: {len(successful)}")
    print(f"处理失败: {len(failed)}")
    print(f"总数据行数: {total_rows:,}")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每个文件耗时: {avg_time_per_file:.2f} 秒")
    print(f"平均处理速度: {total_rows/total_time:.0f} 行/秒" if total_time > 0 else "速度: N/A")
    
    if successful:
        print(f"\n✅ 成功文件已保存到输出目录")
    
    if failed:
        print(f"\n❌ 失败文件列表 ({len(failed)} 个):")
        for i, fail in enumerate(failed, 1):
            print(f"  {i}. {Path(fail['input_file']).name}: {fail['error']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多进程Parquet转JSONL批量转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python parquet_to_jsonl.py /input/parquet/files /output/jsonl/files
  python parquet_to_jsonl.py /input/data /output/result --workers 8
        """
    )
    
    parser.add_argument('input_dir', help='包含Parquet文件的输入目录路径')
    parser.add_argument('output_dir', help='JSONL文件的输出目录路径')
    parser.add_argument('--workers', type=int, default=None, 
                       help='最大工作进程数 (默认: CPU核心数)')
    parser.add_argument('--log', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log))
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    logger.info(f"开始处理: {args.input_dir} -> {args.output_dir}")
    
    # 记录开始时间
    overall_start_time = time.time()
    
    # 处理文件
    results = process_files_parallel(args.input_dir, args.output_dir, args.workers)
    
    # 计算总耗时
    overall_time = time.time() - overall_start_time
    
    # 生成报告
    generate_summary_report(results)
    print(f"\n🏁 全部处理完成，总耗时: {overall_time:.2f} 秒")

if __name__ == "__main__":
    # 设置多进程启动方法（Windows需要，Linux/macOS自动选择最佳方式）
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    
    main()


"""
python convert_parquet2jsonl.py Nemotron-Post-Training-Dataset-v1/ orig2jsonl  --workers 128
"""

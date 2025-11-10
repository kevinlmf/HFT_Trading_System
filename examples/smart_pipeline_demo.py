"""
智能Pipeline演示
展示完整的EDA -> 清理 -> 智能执行流程
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from Execution.engine.pipeline import TradingPipeline, create_sample_data

def main():
    """主演示函数"""
    print("="*80)
    print("HFT智能Pipeline演示")
    print("="*80)
    print("\n本演示将展示完整的处理流程：")
    print("1. EDA - 探索性数据分析")
    print("2. 数据清理 - 自动清理数据")
    print("3. 智能执行 - 根据数据规模选择最优实现")
    print("4. 结果验证 - 验证结果质量")
    print("\n" + "="*80)
    
    # 创建Pipeline
    pipeline = TradingPipeline()
    
    # 测试场景1: 小规模数据（使用Python向量化）
    print("\n" + "="*80)
    print("场景1: 小规模数据 (5,000 订单)")
    print("="*80)
    data1 = create_sample_data(n_records=5000)
    result1 = pipeline.process(data1, data_type="small_orders", auto_clean=True)
    
    # 测试场景2: 中等规模数据（可能使用C++）
    print("\n" + "="*80)
    print("场景2: 中等规模数据 (50,000 订单)")
    print("="*80)
    data2 = create_sample_data(n_records=50000)
    result2 = pipeline.process(data2, data_type="medium_orders", auto_clean=True)
    
    # 测试场景3: 大规模数据（推荐C++）
    print("\n" + "="*80)
    print("场景3: 大规模数据 (500,000 订单)")
    print("="*80)
    data3 = create_sample_data(n_records=500000)
    result3 = pipeline.process(data3, data_type="large_orders", auto_clean=True)
    
    # 打印最终统计
    print("\n" + "="*80)
    print("最终统计")
    print("="*80)
    stats = pipeline.executor.get_stats()
    print(f"总处理订单数: {stats['total_orders']:,}")
    print(f"Python调用: {stats['python_calls']} ({stats['python_ratio']*100:.1f}%)")
    print(f"C++调用: {stats['cpp_calls']} ({stats['cpp_ratio']*100:.1f}%)")
    print(f"CUDA调用: {stats['cuda_calls']} ({stats['cuda_ratio']*100:.1f}%)")
    print("="*80)
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()







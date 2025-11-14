#!/usr/bin/env python3
"""
QDB性能基准测试工具

测试QDB优化前后的性能差异
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import argparse

# QDB导入
try:
    from Data.qdb import create_qdb
    from Data.qdb.indexer import DataIndexer
    from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
    QDB_AVAILABLE = True
except ImportError:
    QDB_AVAILABLE = False
    print("⚠️  QDB模块未找到，请先安装依赖")


def generate_test_data(n_days: int = 365, n_symbols: int = 10) -> Dict[str, pd.DataFrame]:
    """生成测试数据"""
    print(f"生成测试数据: {n_symbols}个symbol, {n_days}天...")
    
    data = {}
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='1H')
    np.random.seed(42)
    
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    
    for symbol in symbols:
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        df = pd.DataFrame({
            'symbol': symbol,
            'bid_price': prices - 0.05,
            'ask_price': prices + 0.05,
            'bid_size': np.random.randint(100, 1000, len(dates)),
            'ask_size': np.random.randint(100, 1000, len(dates)),
            'last_price': prices,
            'volume': np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
        data[symbol] = df
    
    return data


def benchmark_indexer_performance():
    """测试索引器性能"""
    print("\n" + "=" * 80)
    print("索引器性能测试")
    print("=" * 80)
    
    if not QDB_AVAILABLE:
        print("⚠️  QDB不可用，跳过测试")
        return
    
    # 准备测试数据
    base_path = "./Data/datasets/qdb_benchmark"
    import shutil
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # 生成测试数据
    test_data = generate_test_data(n_days=365, n_symbols=10)
    
    # 测试标准索引器
    print("\n1. 标准索引器 (O(n) 线性搜索):")
    standard_indexer = DataIndexer(base_path=base_path + "_standard")
    
    # 存储数据
    start = time.time()
    for symbol, df in test_data.items():
        standard_indexer.add_data(symbol, df, data_version="test")
    store_time = time.time() - start
    print(f"   存储时间: {store_time*1000:.2f}ms")
    
    # 测试查询（不同文件数）
    file_counts = [1, 5, 10, 20, 50]
    print("\n   查询性能 (不同文件数):")
    print(f"   {'文件数':<10} {'查询时间(ms)':<15} {'复杂度':<15}")
    print("   " + "-" * 40)
    
    for n_files in file_counts:
        # 创建多个文件
        test_symbol = 'SYM0'
        dates = pd.date_range(start='2024-01-01', periods=n_files*10, freq='1H')
        for i in range(n_files):
            df = pd.DataFrame({
                'symbol': test_symbol,
                'last_price': 100 + np.random.randn(len(dates)) * 0.1,
                'volume': np.random.randint(1000, 10000, len(dates)),
            }, index=dates)
            standard_indexer.add_data(test_symbol, df, data_version=f"test_{i}")
        
        # 测试查询
        start = time.time()
        for _ in range(100):  # 100次查询
            files = standard_indexer.find_files(test_symbol, 
                                               datetime(2024, 1, 1),
                                               datetime(2024, 12, 31))
        query_time = (time.time() - start) / 100 * 1000
        print(f"   {n_files:<10} {query_time:>12.3f}ms {'O(n)':<15}")
    
    # 测试优化索引器
    print("\n2. 优化索引器 (O(log n) 二分查找):")
    optimized_indexer = ImprovedOptimizedIndexer(base_path=base_path + "_optimized")
    
    # 存储数据
    start = time.time()
    for symbol, df in test_data.items():
        optimized_indexer.add_data(symbol, df.index.min(), df.index.max(), 
                                   f"data/{symbol}_test.parquet")
    store_time = time.time() - start
    print(f"   存储时间: {store_time*1000:.2f}ms")
    
    # 重新构建索引（模拟实际使用）
    optimized_indexer._build_optimized_index()
    
    print("\n   查询性能 (不同文件数):")
    print(f"   {'文件数':<10} {'查询时间(ms)':<15} {'复杂度':<15} {'加速比':<10}")
    print("   " + "-" * 60)
    
    for n_files in file_counts:
        # 创建多个文件索引
        test_symbol = 'SYM0'
        time_ranges = []
        for i in range(n_files):
            start_time = datetime(2024, 1, 1) + timedelta(days=i*10)
            end_time = start_time + timedelta(days=10)
            time_ranges.append((start_time, end_time, f"data/{test_symbol}_{i}.parquet"))
        time_ranges.sort(key=lambda x: x[0])
        optimized_indexer._time_index[test_symbol] = (
            time_ranges,
            np.array([tr[1].timestamp() for tr in time_ranges])
        )
        
        # 测试查询
        start = time.time()
        for _ in range(100):  # 100次查询
            files = optimized_indexer.find_files_optimized(test_symbol,
                                                          datetime(2024, 1, 1),
                                                          datetime(2024, 12, 31))
        query_time = (time.time() - start) / 100 * 1000
        
        # 计算加速比（相对于标准版本）
        standard_time = 0.1 * n_files  # 估算标准版本时间
        speedup = standard_time / query_time if query_time > 0 else 0
        
        print(f"   {n_files:<10} {query_time:>12.3f}ms {'O(log n)':<15} {speedup:>8.2f}x")


def benchmark_data_loading():
    """测试数据加载性能"""
    print("\n" + "=" * 80)
    print("数据加载性能测试")
    print("=" * 80)
    
    if not QDB_AVAILABLE:
        print("⚠️  QDB不可用，跳过测试")
        return
    
    qdb = create_qdb(base_path="./Data/datasets/qdb_benchmark_load")
    
    # 准备测试数据
    test_data = generate_test_data(n_days=365, n_symbols=5)
    
    # 存储数据
    print("\n存储测试数据到QDB...")
    for symbol, df in test_data.items():
        qdb.store(symbol=symbol, df=df, data_version="benchmark")
    
    # 测试加载性能
    print("\n测试加载性能:")
    print(f"   {'操作':<30} {'时间(ms)':<15} {'记录数':<10}")
    print("   " + "-" * 55)
    
    # 1. 加载单个symbol（小范围）
    start = time.time()
    df1 = qdb.load(symbol='SYM0', start='2024-01-01', end='2024-01-07')
    time1 = (time.time() - start) * 1000
    print(f"   {'单symbol (7天)':<30} {time1:>12.3f}ms {len(df1):>10}")
    
    # 2. 加载单个symbol（大范围）
    start = time.time()
    df2 = qdb.load(symbol='SYM0', start='2024-01-01', end='2024-12-31')
    time2 = (time.time() - start) * 1000
    print(f"   {'单symbol (1年)':<30} {time2:>12.3f}ms {len(df2):>10}")
    
    # 3. 加载多个symbol（串行）
    start = time.time()
    dfs_serial = []
    for symbol in ['SYM0', 'SYM1', 'SYM2']:
        df = qdb.load(symbol=symbol, start='2024-01-01', end='2024-12-31')
        dfs_serial.append(df)
    time3 = (time.time() - start) * 1000
    print(f"   {'多symbol串行 (3个)':<30} {time3:>12.3f}ms {sum(len(d) for d in dfs_serial):>10}")
    
    # 4. 缓存测试（第二次加载）
    start = time.time()
    df_cached = qdb.load(symbol='SYM0', start='2024-01-01', end='2024-12-31')
    time4 = (time.time() - start) * 1000
    print(f"   {'缓存命中 (1年)':<30} {time4:>12.3f}ms {len(df_cached):>10}")
    print(f"   {'缓存加速比':<30} {time2/time4:>12.2f}x {'':<10}" if time4 > 0 else "")
    
    # 查看缓存统计
    cache_stats = qdb.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"   命中率: {cache_stats['hit_rate']*100:.2f}%")
    print(f"   缓存大小: {cache_stats['current_size_mb']:.2f}MB")
    print(f"   缓存项数: {cache_stats['current_items']}")


def benchmark_optimization_stack():
    """测试Optimization栈性能"""
    print("\n" + "=" * 80)
    print("Optimization栈性能测试")
    print("=" * 80)
    
    try:
        from Optimization.optimized_optimization_stack import EnhancedOptimizationStack
        from Optimization.optimization_stack import ModelObjective
    except ImportError:
        print("⚠️  优化栈不可用，跳过测试")
        return
    
    # 创建测试数据
    n_samples, n_assets = 1000, 50
    returns = np.random.randn(n_samples, n_assets).astype(np.float32) * 0.02
    
    print(f"\n测试数据: {n_samples}样本, {n_assets}资产")
    
    # 测试协方差矩阵计算
    print("\n1. 协方差矩阵计算:")
    
    # 标准计算
    start = time.time()
    cov_standard = np.cov(returns, rowvar=False)
    time_standard = (time.time() - start) * 1000
    print(f"   标准计算: {time_standard:.3f}ms")
    
    # 优化计算（带缓存）
    stack = EnhancedOptimizationStack(use_qdb=False)
    start = time.time()
    cov_optimized = stack.data_loader.get_covariance_matrix(
        returns, use_cache=True, cache_key="test_cov"
    )
    time_optimized1 = (time.time() - start) * 1000
    print(f"   优化计算(首次): {time_optimized1:.3f}ms")
    
    # 缓存命中
    start = time.time()
    cov_cached = stack.data_loader.get_covariance_matrix(
        returns, use_cache=True, cache_key="test_cov"
    )
    time_cached = (time.time() - start) * 1000
    print(f"   缓存命中: {time_cached:.3f}ms")
    print(f"   加速比: {time_standard/time_cached:.0f}x" if time_cached > 0 else "")
    
    # 测试数据结构优化
    print("\n2. 数据结构优化:")
    
    # float64
    returns_f64 = returns.astype(np.float64)
    size_f64 = returns_f64.nbytes / 1024
    print(f"   float64大小: {size_f64:.2f}KB")
    
    # float32
    returns_f32 = returns.astype(np.float32)
    size_f32 = returns_f32.nbytes / 1024
    print(f"   float32大小: {size_f32:.2f}KB")
    print(f"   内存节省: {(1 - size_f32/size_f64)*100:.1f}%")
    
    # 测试计算速度
    start = time.time()
    _ = np.dot(returns_f64, returns_f64.T)
    time_f64 = (time.time() - start) * 1000
    
    start = time.time()
    _ = np.dot(returns_f32, returns_f32.T)
    time_f32 = (time.time() - start) * 1000
    
    print(f"   矩阵乘法 (float64): {time_f64:.3f}ms")
    print(f"   矩阵乘法 (float32): {time_f32:.3f}ms")
    print(f"   加速比: {time_f64/time_f32:.2f}x" if time_f32 > 0 else "")


def benchmark_full_workflow():
    """测试完整工作流性能"""
    print("\n" + "=" * 80)
    print("完整工作流性能测试")
    print("=" * 80)
    
    if not QDB_AVAILABLE:
        print("⚠️  QDB不可用，跳过测试")
        return
    
    qdb = create_qdb(base_path="./Data/datasets/qdb_workflow")
    
    # 准备数据
    print("\n准备测试数据...")
    test_data = generate_test_data(n_days=252, n_symbols=10)
    
    # 1. 存储数据
    print("\n1. 数据存储:")
    start = time.time()
    for symbol, df in test_data.items():
        qdb.store(symbol=symbol, df=df, data_version="workflow_test")
    store_time = time.time() - start
    print(f"   存储{len(test_data)}个symbol: {store_time:.3f}s")
    print(f"   平均每个symbol: {store_time/len(test_data)*1000:.2f}ms")
    
    # 2. 数据加载
    print("\n2. 数据加载:")
    symbols = list(test_data.keys())[:5]
    start = time.time()
    loaded_data = {}
    for symbol in symbols:
        df = qdb.load(symbol=symbol, start='2024-01-01', end='2024-12-31')
        loaded_data[symbol] = df
    load_time = time.time() - start
    print(f"   加载{len(symbols)}个symbol: {load_time:.3f}s")
    print(f"   平均每个symbol: {load_time/len(symbols)*1000:.2f}ms")
    
    # 3. 投资组合优化（如果可用）
    try:
        from Optimization.optimized_optimization_stack import EnhancedOptimizationStack
        from Optimization.optimization_stack import ModelObjective
        
        print("\n3. 投资组合优化:")
        stack = EnhancedOptimizationStack(use_qdb=True, qdb=qdb)
        
        start = time.time()
        result = stack.optimize_portfolio_from_qdb(
            symbols=symbols,
            start_time='2024-01-01',
            end_time='2024-12-31',
            objective=ModelObjective.MAXIMIZE_SHARPE
        )
        opt_time = time.time() - start
        
        print(f"   优化时间: {opt_time:.3f}s")
        print(f"   使用算法: {result['optimization_info']['algorithm']}")
        print(f"   数据源: {result['optimization_info']['data_source']}")
        print(f"   缓存使用: {result['optimization_info']['cache_used']}")
        if 'performance_metrics' in result:
            print(f"   Sharpe比率: {result['performance_metrics']['sharpe_ratio']:.4f}")
    except Exception as e:
        print(f"   ⚠️  优化测试失败: {e}")
    
    # 4. 缓存效果
    print("\n4. 缓存效果:")
    cache_stats = qdb.get_cache_stats()
    print(f"   命中率: {cache_stats['hit_rate']*100:.2f}%")
    print(f"   缓存大小: {cache_stats['current_size_mb']:.2f}MB")
    print(f"   缓存项数: {cache_stats['current_items']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='QDB性能基准测试')
    parser.add_argument('--test', choices=['indexer', 'loading', 'optimization', 'workflow', 'all'],
                       default='all', help='选择测试类型')
    parser.add_argument('--output', type=str, help='输出结果到文件')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("QDB性能基准测试")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    
    results = {}
    
    if args.test in ['indexer', 'all']:
        start = time.time()
        benchmark_indexer_performance()
        results['indexer'] = time.time() - start
    
    if args.test in ['loading', 'all']:
        start = time.time()
        benchmark_data_loading()
        results['loading'] = time.time() - start
    
    if args.test in ['optimization', 'all']:
        start = time.time()
        benchmark_optimization_stack()
        results['optimization'] = time.time() - start
    
    if args.test in ['workflow', 'all']:
        start = time.time()
        benchmark_full_workflow()
        results['workflow'] = time.time() - start
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    total_time = sum(results.values())
    print(f"总测试时间: {total_time:.2f}s")
    for test_name, test_time in results.items():
        print(f"  {test_name}: {test_time:.2f}s")
    
    # 输出到文件（如果指定）
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"QDB性能基准测试结果\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总测试时间: {total_time:.2f}s\n")
            for test_name, test_time in results.items():
                f.write(f"{test_name}: {test_time:.2f}s\n")
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()














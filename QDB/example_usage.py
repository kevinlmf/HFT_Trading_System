"""
QDB使用示例

展示如何使用QDB系统进行数据存储、检索和版本管理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from QDB import QDB, create_qdb
from QDB.ingestion import HistoricalDownloader


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("QDB基本使用示例")
    print("=" * 80)
    
    # 1. 初始化QDB
    qdb = create_qdb(base_path="./Data/datasets/qdb", memmap=True)
    
    # 2. 创建示例数据
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'symbol': 'SPY',
        'bid_price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
        'ask_price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1) + 0.05,
        'bid_size': np.random.randint(100, 1000, len(dates)),
        'ask_size': np.random.randint(100, 1000, len(dates)),
        'last_price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
        'volume': np.random.randint(1000, 10000, len(dates)),
    }, index=dates)
    
    # 3. 存储数据
    print("\n1. 存储数据到QDB...")
    file_path = qdb.store(
        symbol="SPY",
        df=df,
        data_version="qdb_2024Q1",
        experiment_id="RL_v3",
        feature_version="features_v7",
        description="SPY hourly data for January 2024",
        tags=["equity", "hourly", "test"]
    )
    print(f"   存储路径: {file_path}")
    
    # 4. 快速加载数据（目标 < 10ms）
    print("\n2. 快速加载数据...")
    import time
    start = time.time()
    loaded_df = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-05")
    load_time = (time.time() - start) * 1000
    print(f"   加载了 {len(loaded_df)} 条记录，耗时 {load_time:.2f}ms")
    print(f"   数据范围: {loaded_df.index.min()} 到 {loaded_df.index.max()}")
    
    # 5. RL训练采样
    print("\n3. RL训练采样...")
    sample_df = qdb.sample(symbol="SPY", window=100)
    print(f"   采样了 {len(sample_df)} 条记录")
    
    # 6. 查看缓存统计
    print("\n4. 缓存统计...")
    cache_stats = qdb.get_cache_stats()
    print(f"   命中率: {cache_stats['hit_rate']*100:.2f}%")
    print(f"   缓存大小: {cache_stats['current_size_mb']:.2f}MB")
    
    # 7. 查看版本信息
    print("\n5. 版本信息...")
    versions = qdb.list_versions(symbol="SPY")
    for version in versions:
        print(f"   版本ID: {version.version_id}")
        print(f"   数据版本: {version.data_version}")
        print(f"   实验ID: {version.experiment_id}")
        print(f"   创建时间: {version.created_at}")
    
    # 8. 查看QDB信息
    print("\n6. QDB信息...")
    info = qdb.get_info()
    print(f"   存储路径: {info['base_path']}")
    print(f"   Symbol数量: {info['n_symbols']}")
    print(f"   Symbols: {info['symbols']}")


def example_historical_download():
    """历史数据下载示例"""
    print("\n" + "=" * 80)
    print("历史数据下载示例")
    print("=" * 80)
    
    qdb = create_qdb()
    downloader = HistoricalDownloader(qdb)
    
    # 模拟从外部数据源下载
    def download_func(symbol: str) -> pd.DataFrame:
        """模拟下载函数"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
        np.random.seed(hash(symbol) % 1000)
        
        return pd.DataFrame({
            'symbol': symbol,
            'open': 100 + np.random.randn(len(dates)) * 5,
            'high': 105 + np.random.randn(len(dates)) * 5,
            'low': 95 + np.random.randn(len(dates)) * 5,
            'close': 100 + np.random.randn(len(dates)) * 5,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
    
    # 批量下载
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    results = downloader.download_batch(
        symbols=symbols,
        download_func=download_func,
        data_version="historical_2024Q1",
        description="Historical daily data"
    )
    
    print(f"\n下载结果:")
    for symbol, file_path in results.items():
        if file_path:
            print(f"  {symbol}: {file_path}")
        else:
            print(f"  {symbol}: 下载失败")


def example_multi_strategy_training():
    """多策略并行训练示例"""
    print("\n" + "=" * 80)
    print("多策略并行训练示例（模拟）")
    print("=" * 80)
    
    qdb = create_qdb()
    
    # 模拟多个RL agent同时训练
    symbols = ['SPY', 'AAPL', 'MSFT']
    
    # 为每个symbol创建数据
    for symbol in symbols:
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        np.random.seed(hash(symbol) % 1000)
        
        df = pd.DataFrame({
            'symbol': symbol,
            'last_price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
            'volume': np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
        
        qdb.store(
            symbol=symbol,
            df=df,
            data_version="qdb_2024Q1",
            experiment_id="multi_strategy_training"
        )
    
    # 模拟多个agent同时采样
    print("\n模拟3个RL agent同时训练:")
    for i, symbol in enumerate(symbols):
        sample = qdb.sample(symbol=symbol, window=1000)
        print(f"  Agent {i+1} ({symbol}): 采样了 {len(sample)} 条记录")
    
    # 查看缓存效果
    cache_stats = qdb.get_cache_stats()
    print(f"\n缓存统计:")
    print(f"  命中率: {cache_stats['hit_rate']*100:.2f}%")
    print(f"  缓存项数: {cache_stats['current_items']}")


def example_data_validation():
    """数据验证示例"""
    print("\n" + "=" * 80)
    print("数据验证示例")
    print("=" * 80)
    
    qdb = create_qdb()
    
    # 验证数据
    is_valid, errors = qdb.validate_data("SPY", start="2024-01-01", end="2024-01-05")
    
    if is_valid:
        print("✓ 数据验证通过")
    else:
        print("✗ 数据验证失败:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_historical_download()
    example_multi_strategy_training()
    example_data_validation()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)


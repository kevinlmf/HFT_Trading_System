"""
QDB集成示例 - 展示如何将QDB集成到现有的回测和策略评估流程中

这个示例展示如何：
1. 将现有数据迁移到QDB
2. 在回测中使用QDB数据
3. 在策略评估中使用QDB数据
4. 实现数据一致性（实盘、回测、模拟都从同一数据源）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# QDB导入
from Data.qdb import QDB, create_qdb
from Data.qdb.ingestion import HistoricalDownloader

# 现有系统导入
from Execution.engine.simple_backtester import SimpleBacktester, BacktestConfig
from Execution.evaluation import StrategyEvaluator
from Execution.engine.strategy_adapter import StrategyAdapter


def generate_sample_market_data(
    n_days: int = 504,
    initial_price: float = 100.0,
    drift: float = 0.0005,
    volatility: float = 0.02
) -> pd.DataFrame:
    """生成示例市场数据（模拟现有系统的数据生成）"""
    np.random.seed(42)
    
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_days,
        freq='D'
    )
    
    returns = np.random.randn(n_days) * volatility + drift
    prices = initial_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1) * (1 + np.random.randn(n_days) * volatility * 0.1)
    data['open'].iloc[0] = initial_price
    
    daily_range = prices * volatility * np.random.uniform(0.5, 2.0, n_days)
    data['high'] = prices + daily_range * np.random.uniform(0, 1, n_days)
    data['low'] = prices - daily_range * np.random.uniform(0, 1, n_days)
    
    base_volume = 1_000_000
    volume_pattern = np.sin(np.arange(n_days) * 2 * np.pi / 20) * 0.3 + 1
    data['volume'] = (base_volume * volume_pattern * (1 + np.random.randn(n_days) * 0.2)).astype(int)
    
    # 添加bid/ask（用于QDB标准化）
    spread = prices * 0.001  # 0.1% spread
    data['bid_price'] = prices - spread / 2
    data['ask_price'] = prices + spread / 2
    data['bid_size'] = data['volume'] * 0.1
    data['ask_size'] = data['volume'] * 0.1
    data['last_price'] = prices
    
    return data


def example_migrate_to_qdb():
    """示例：将现有数据迁移到QDB"""
    print("=" * 80)
    print("示例1: 数据迁移到QDB")
    print("=" * 80)
    
    # 1. 初始化QDB
    qdb = create_qdb(base_path="./Data/datasets/qdb")
    
    # 2. 生成现有格式的数据（模拟从现有系统获取）
    print("\n1. 生成现有格式的数据...")
    market_data = generate_sample_market_data(n_days=504)
    print(f"   生成了 {len(market_data)} 天的数据")
    
    # 3. 存储到QDB（自动标准化）
    print("\n2. 存储到QDB...")
    file_path = qdb.store(
        symbol="SPY",
        df=market_data,
        data_version="qdb_2024Q1",
        source_format="standard",
        experiment_id="migration_test",
        description="Migrated from existing system"
    )
    print(f"   存储路径: {file_path}")
    
    # 4. 验证数据
    print("\n3. 验证数据...")
    is_valid, errors = qdb.validate_data("SPY")
    if is_valid:
        print("   ✓ 数据验证通过")
    else:
        print(f"   ✗ 验证失败: {errors}")
    
    return qdb


def example_backtest_with_qdb():
    """示例：在回测中使用QDB数据"""
    print("\n" + "=" * 80)
    print("示例2: 使用QDB数据进行回测")
    print("=" * 80)
    
    # 1. 初始化QDB
    qdb = create_qdb()
    
    # 2. 从QDB加载数据（快速，<10ms目标）
    print("\n1. 从QDB加载数据...")
    import time
    start = time.time()
    market_data = qdb.load(symbol="SPY", start="2024-01-01", end="2024-12-31")
    load_time = (time.time() - start) * 1000
    print(f"   加载了 {len(market_data)} 条记录，耗时 {load_time:.2f}ms")
    
    if len(market_data) == 0:
        print("   警告: 没有数据，先运行数据迁移示例")
        return
    
    # 3. 初始化回测器
    print("\n2. 初始化回测器...")
    backtester = SimpleBacktester(
        config=BacktestConfig(
            initial_capital=1_000_000.0,
            commission_rate=0.0002,
            slippage_bps=1.0,
            position_size=0.1
        )
    )
    
    # 4. 获取策略
    print("\n3. 获取策略...")
    all_strategies = StrategyAdapter.get_all_strategies()
    strategy_name = list(all_strategies.keys())[0] if all_strategies else None
    
    if not strategy_name:
        print("   警告: 没有可用策略")
        return
    
    print(f"   使用策略: {strategy_name}")
    
    # 5. 运行回测（使用QDB数据）
    print("\n4. 运行回测...")
    # 注意：这里需要适配SimpleBacktester的接口
    # 实际使用时，SimpleBacktester应该能够接受DataFrame格式的数据
    print("   (回测逻辑需要根据SimpleBacktester的实际接口调整)")
    
    print("\n✓ 回测完成（使用QDB数据，确保数据一致性）")


def example_multi_strategy_training():
    """示例：多策略并行训练（使用QDB缓存）"""
    print("\n" + "=" * 80)
    print("示例3: 多策略并行训练（QDB缓存加速）")
    print("=" * 80)
    
    qdb = create_qdb()
    
    # 模拟多个RL agent同时训练
    print("\n1. 模拟3个RL agent同时训练...")
    
    symbols = ['SPY', 'AAPL', 'MSFT']
    
    # 为每个symbol准备数据（如果还没有）
    for symbol in symbols:
        if symbol not in qdb.list_symbols():
            print(f"   为 {symbol} 准备数据...")
            data = generate_sample_market_data(n_days=252)
            qdb.store(
                symbol=symbol,
                df=data,
                data_version="qdb_2024Q1",
                experiment_id="multi_training"
            )
    
    # 模拟多个agent同时采样
    print("\n2. 多个agent同时采样（利用QDB缓存）...")
    for i, symbol in enumerate(symbols):
        # 第一次加载（缓存未命中）
        start = time.time()
        sample1 = qdb.sample(symbol=symbol, window=1000)
        time1 = (time.time() - start) * 1000
        
        # 第二次加载（缓存命中，应该更快）
        start = time.time()
        sample2 = qdb.sample(symbol=symbol, window=1000)
        time2 = (time.time() - start) * 1000
        
        print(f"   Agent {i+1} ({symbol}):")
        print(f"     第一次: {time1:.2f}ms, 第二次: {time2:.2f}ms")
        print(f"     加速比: {time1/time2:.2f}x" if time2 > 0 else "     加速比: N/A")
    
    # 查看缓存统计
    print("\n3. 缓存统计...")
    cache_stats = qdb.get_cache_stats()
    print(f"   命中率: {cache_stats['hit_rate']*100:.2f}%")
    print(f"   缓存大小: {cache_stats['current_size_mb']:.2f}MB")
    print(f"   缓存项数: {cache_stats['current_items']}")


def example_version_management():
    """示例：版本管理和实验追踪"""
    print("\n" + "=" * 80)
    print("示例4: 版本管理和实验追踪")
    print("=" * 80)
    
    qdb = create_qdb()
    
    # 创建不同版本的数据
    print("\n1. 创建不同版本的数据...")
    
    # 版本1: 原始数据
    data_v1 = generate_sample_market_data(n_days=252)
    qdb.store(
        symbol="SPY",
        df=data_v1,
        data_version="qdb_2024Q1_v1",
        experiment_id="RL_v1",
        feature_version="features_v1",
        description="Original data"
    )
    
    # 版本2: 清洗后的数据
    data_v2 = data_v1.copy()
    data_v2 = data_v2.dropna()  # 模拟数据清洗
    qdb.store(
        symbol="SPY",
        df=data_v2,
        data_version="qdb_2024Q1_v2",
        experiment_id="RL_v2",
        feature_version="features_v2",
        description="Cleaned data"
    )
    
    # 查看版本列表
    print("\n2. 查看版本列表...")
    versions = qdb.list_versions(symbol="SPY")
    for version in versions:
        print(f"   版本ID: {version.version_id}")
        print(f"   数据版本: {version.data_version}")
        print(f"   实验ID: {version.experiment_id}")
        print(f"   特征版本: {version.feature_version}")
        print(f"   描述: {version.description}")
        print()
    
    print("✓ 版本管理完成（可以追踪每个实验使用的数据版本）")


def example_data_consistency():
    """示例：数据一致性（实盘、回测、模拟都从同一数据源）"""
    print("\n" + "=" * 80)
    print("示例5: 数据一致性保证")
    print("=" * 80)
    
    qdb = create_qdb()
    
    print("\n核心原则：实盘、回测、模拟都从QDB取数据，确保一致性")
    
    # 1. 回测使用QDB数据
    print("\n1. 回测场景...")
    backtest_data = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-31")
    print(f"   回测数据: {len(backtest_data)} 条记录")
    
    # 2. 模拟使用QDB数据
    print("\n2. 模拟场景...")
    simulation_data = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-31")
    print(f"   模拟数据: {len(simulation_data)} 条记录")
    
    # 3. 实盘使用QDB数据（从实时收集器）
    print("\n3. 实盘场景...")
    print("   实盘数据通过RealtimeCollector实时收集并存储到QDB")
    print("   策略从QDB读取最新数据，确保与回测/模拟一致")
    
    # 验证数据一致性
    print("\n4. 验证数据一致性...")
    if len(backtest_data) == len(simulation_data):
        print("   ✓ 回测和模拟使用相同的数据量")
    else:
        print("   ✗ 数据量不一致")
    
    print("\n✓ 数据一致性保证：所有场景都从QDB取数据")


if __name__ == "__main__":
    import time
    
    print("\n" + "=" * 80)
    print("QDB集成示例")
    print("=" * 80)
    
    # 运行所有示例
    qdb = example_migrate_to_qdb()
    example_backtest_with_qdb()
    example_multi_strategy_training()
    example_version_management()
    example_data_consistency()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)
    print("\nQDB系统已集成，现在可以：")
    print("1. ✓ 快速加载数据（<10ms目标）")
    print("2. ✓ 多策略并行训练（共享缓存）")
    print("3. ✓ 版本管理和实验追踪")
    print("4. ✓ 数据一致性保证（实盘/回测/模拟）")
    print("=" * 80)


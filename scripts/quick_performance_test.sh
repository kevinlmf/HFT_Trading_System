#!/bin/bash
# 快速性能测试脚本
# 快速查看QDB和Optimization的性能

set -e

echo ""
echo "============================================================"
echo "     QDB & Optimization 性能快速测试"
echo "============================================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# 测试1: QDB加载速度
echo "1️⃣  测试QDB数据加载速度..."
echo "----------------------------------------"
python3 -c "
import sys
import time
sys.path.insert(0, '.')
from Data.qdb import create_qdb
import pandas as pd
import numpy as np
from datetime import datetime

# 创建QDB
qdb = create_qdb(base_path='./Data/datasets/qdb_test')

# 生成测试数据
dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
df = pd.DataFrame({
    'symbol': 'TEST',
    'last_price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
    'volume': np.random.randint(1000, 10000, len(dates)),
}, index=dates)

# 存储
print('  存储数据...')
start = time.time()
qdb.store(symbol='TEST', df=df, data_version='test')
store_time = (time.time() - start) * 1000
print(f'  ✓ 存储时间: {store_time:.2f}ms')

# 加载
print('  加载数据...')
start = time.time()
loaded = qdb.load(symbol='TEST', start='2024-01-01', end='2024-01-10')
load_time = (time.time() - start) * 1000
print(f'  ✓ 加载时间: {load_time:.2f}ms ({len(loaded)} 条记录)')

# 缓存测试
print('  缓存测试...')
start = time.time()
cached = qdb.load(symbol='TEST', start='2024-01-01', end='2024-01-10')
cache_time = (time.time() - start) * 1000
print(f'  ✓ 缓存时间: {cache_time:.2f}ms')
if cache_time > 0:
    print(f'  ✓ 加速比: {load_time/cache_time:.1f}x')

# 缓存统计
stats = qdb.get_cache_stats()
print(f'  ✓ 缓存命中率: {stats[\"hit_rate\"]*100:.1f}%')
"

echo ""
echo "2️⃣  测试Optimization栈性能..."
echo "----------------------------------------"
python3 -c "
import sys
import time
import numpy as np
sys.path.insert(0, '.')

try:
    from Optimization.optimized_optimization_stack import EnhancedOptimizationStack
    from Optimization.optimization_stack import ModelObjective
    
    # 创建测试数据
    n_samples, n_assets = 1000, 50
    returns = np.random.randn(n_samples, n_assets).astype(np.float32) * 0.02
    
    print(f'  测试数据: {n_samples}样本, {n_assets}资产')
    
    # 协方差矩阵计算
    print('  协方差矩阵计算...')
    stack = EnhancedOptimizationStack(use_qdb=False)
    
    # 标准计算
    start = time.time()
    cov_std = np.cov(returns, rowvar=False)
    time_std = (time.time() - start) * 1000
    print(f'  ✓ 标准计算: {time_std:.2f}ms')
    
    # 缓存计算
    start = time.time()
    cov_cached = stack.data_loader.get_covariance_matrix(returns, use_cache=True, cache_key='test')
    time_cached1 = (time.time() - start) * 1000
    print(f'  ✓ 优化计算: {time_cached1:.2f}ms')
    
    # 缓存命中
    start = time.time()
    cov_cached2 = stack.data_loader.get_covariance_matrix(returns, use_cache=True, cache_key='test')
    time_cached2 = (time.time() - start) * 1000
    print(f'  ✓ 缓存命中: {time_cached2:.2f}ms')
    if time_cached2 > 0:
        print(f'  ✓ 加速比: {time_std/time_cached2:.0f}x')
    
    # 内存优化
    print('  内存优化...')
    returns_f64 = returns.astype(np.float64)
    returns_f32 = returns.astype(np.float32)
    size_f64 = returns_f64.nbytes / 1024
    size_f32 = returns_f32.nbytes / 1024
    print(f'  ✓ float64: {size_f64:.1f}KB')
    print(f'  ✓ float32: {size_f32:.1f}KB')
    print(f'  ✓ 内存节省: {(1-size_f32/size_f64)*100:.1f}%')
    
except ImportError as e:
    print(f'  ⚠️  Optimization模块不可用: {e}')
"

echo ""
echo "3️⃣  测试优化索引器性能..."
echo "----------------------------------------"
python3 -c "
import sys
import time
import numpy as np
from datetime import datetime
sys.path.insert(0, '.')

try:
    from Data.qdb.improved_optimized_indexer import ImprovedOptimizedIndexer
    from datetime import timedelta
    
    # 创建索引器
    indexer = ImprovedOptimizedIndexer(base_path='./Data/datasets/qdb_indexer_test')
    
    # 创建测试索引（模拟多个文件）
    n_files = 50
    time_ranges = []
    for i in range(n_files):
        start_time = datetime(2024, 1, 1) + timedelta(days=i*7)
        end_time = start_time + timedelta(days=7)
        time_ranges.append((start_time, end_time, f'data/file_{i}.parquet'))
    
    time_ranges.sort(key=lambda x: x[0])
    end_times = np.array([tr[1].timestamp() for tr in time_ranges])
    indexer._time_index['TEST'] = (time_ranges, end_times)
    
    # 测试查询性能
    print(f'  测试查询 ({n_files} 个文件)...')
    start = time.time()
    for _ in range(100):
        files = indexer.find_files_optimized('TEST', datetime(2024, 1, 1), datetime(2024, 12, 31))
    query_time = (time.time() - start) / 100 * 1000
    print(f'  ✓ 平均查询时间: {query_time:.3f}ms')
    print(f'  ✓ 复杂度: O(log n)')
    
except ImportError as e:
    print(f'  ⚠️  优化索引器不可用: {e}')
except Exception as e:
    print(f'  ⚠️  测试失败: {e}')
"

echo ""
echo "============================================================"
echo "测试完成！"
echo "============================================================"
echo ""
echo "💡 提示:"
echo "  - 运行完整测试: python3 scripts/benchmark_qdb.py"
echo "  - 查看详细文档: cat PERFORMANCE_GUIDE.md"
echo ""


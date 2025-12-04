# QDB快速开始指南

## 🚀 5分钟快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

确保安装了：
- `pyarrow>=12.0.0` (Parquet支持)
- `fastparquet>=2023.0.0` (备用引擎)

### 2. 基本使用

```python
from Data.qdb import create_qdb
import pandas as pd

# 初始化QDB
qdb = create_qdb(base_path="./Data/datasets/qdb")

# 准备数据（标准格式）
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
df = pd.DataFrame({
    'symbol': 'SPY',
    'bid_price': [100.0] * len(dates),
    'ask_price': [100.05] * len(dates),
    'last_price': [100.0] * len(dates),
    'volume': [1000] * len(dates),
}, index=dates)

# 存储数据
qdb.store(
    symbol="SPY",
    df=df,
    data_version="qdb_2024Q1"
)

# 快速加载（<10ms目标）
df_loaded = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-05")
print(f"加载了 {len(df_loaded)} 条记录")
```

### 3. 运行示例

```bash
# 基本使用示例
python Data/qdb/example_usage.py

# 集成示例（展示如何与现有系统集成）
python Data/qdb/integration_example.py
```

## 📚 核心功能

### 存储数据
```python
qdb.store(
    symbol="SPY",
    df=dataframe,
    data_version="qdb_2024Q1",
    experiment_id="RL_v3",
    feature_version="features_v7"
)
```

### 快速加载
```python
# 时间范围查询（<10ms目标）
df = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-02")
```

### RL训练采样
```python
# 随机采样（利用缓存）
batch = qdb.sample(symbol="AAPL", window=1000)
```

### 版本管理
```python
# 查询版本
versions = qdb.list_versions(symbol="SPY", experiment_id="RL_v3")

# 获取最新版本
latest = qdb.get_latest_version(symbol="SPY")
```

### 缓存统计
```python
stats = qdb.get_cache_stats()
print(f"命中率: {stats['hit_rate']*100:.2f}%")
```

## 🔧 高级配置

### 自定义缓存配置
```python
from Data.qdb import QDB, CacheConfig

cache_config = CacheConfig(
    max_size_mb=2048,      # 最大缓存2GB
    max_items=200,         # 最多200个缓存项
    ttl_seconds=7200,      # 2小时过期
    eviction_policy="LRU"  # LRU淘汰策略
)

qdb = QDB(cache_config=cache_config)
```

### 实时数据收集
```python
from Data.qdb.ingestion import RealtimeCollector
from Data.connectors import AlpacaConnector

connector = AlpacaConnector(api_key="...", api_secret="...")
qdb = create_qdb()
collector = RealtimeCollector(connector, qdb, buffer_size=1000)

await collector.start(['AAPL', 'MSFT'])
```

### 历史数据下载
```python
from Data.qdb.ingestion import HistoricalDownloader

downloader = HistoricalDownloader(qdb)

# 从文件下载
downloader.download_from_file(
    symbol="SPY",
    file_path="data.csv",
    data_version="historical_2024Q1"
)

# 批量下载
def download_func(symbol):
    # 你的下载逻辑
    return dataframe

results = downloader.download_batch(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    download_func=download_func
)
```

## 📖 更多文档

- **架构文档**: `Data/qdb/README.md`
- **实现总结**: `Data/qdb/IMPLEMENTATION_SUMMARY.md`
- **使用示例**: `Data/qdb/example_usage.py`
- **集成示例**: `Data/qdb/integration_example.py`

## 🎯 核心优势

1. **一致性**: 实盘、回测、模拟都从同一数据源
2. **高性能**: Parquet + 索引 + 缓存，加载<10ms
3. **可复现**: 完整的版本管理和实验追踪
4. **可扩展**: 支持多策略并行训练，共享缓存

## 💡 最佳实践

1. **数据标准化**: 所有数据源都通过QDB存储，确保格式一致
2. **版本管理**: 每个实验都记录数据版本和特征版本
3. **缓存利用**: 多策略训练时，利用缓存加速
4. **定期验证**: 使用 `validate_data()` 确保数据完整性

## 🐛 常见问题

### Q: 如何迁移现有数据到QDB？
A: 使用 `HistoricalDownloader` 或直接调用 `qdb.store()`，QDB会自动标准化格式。

### Q: 如何确保数据一致性？
A: 所有模块（回测、模拟、实盘）都从QDB取数据，确保使用相同的数据源。

### Q: 缓存命中率低怎么办？
A: 增加 `max_size_mb` 或 `max_items`，或者检查数据访问模式。

### Q: 如何追踪实验使用的数据版本？
A: 使用 `qdb.list_versions()` 查询，每个版本都记录了实验ID和数据版本。

## 📞 支持

如有问题，请查看：
1. 示例代码：`example_usage.py` 和 `integration_example.py`
2. 文档：`README.md` 和 `IMPLEMENTATION_SUMMARY.md`
3. 代码注释：所有模块都有详细的文档字符串


"""
QDB系统架构文档

QDB (Quantitative Database) 是HFT系统的核心数据层，提供：
1. 一致性（Consistency）- 实盘、回测、模拟都从同一数据源取数据
2. 可复现性（Reproducibility）- 任何训练/实验都能复现
3. 可扩展性（Scalability）- 数据量大时仍能快速读写、支持多策略并行

架构图：

┌──────────────────────────────┐
│      Data Sources            │
│ (Binance / Polygon / Yahoo)  │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│   Data Ingestion Layer       │
│  - RealtimeCollector         │
│  - HistoricalDownloader      │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│         QDB Layer            │
│  ┌────────────────────────┐  │
│  │  Schema (标准化)      │  │
│  │  - normalize_dataframe │  │
│  │  - validate_schema     │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  Indexer (索引)        │  │
│  │  - Parquet存储         │  │
│  │  - 时间范围查询        │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  Cache (缓存)          │  │
│  │  - LRU淘汰策略         │  │
│  │  - 内存映射            │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  Versioning (版本)     │  │
│  │  - 实验追踪            │  │
│  │  - 数据校验和          │  │
│  └────────────────────────┘  │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│     Feature Layer            │
│  - factor_extractor.py       │
│  - rolling_features.py       │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  RL / Strategy Layer         │
│  - PPO / SAC / BC Agents     │
│  - train_ppo.py              │
└──────────────────────────────┘

核心API：

1. 存储数据
   qdb = QDB()
   qdb.store(symbol="SPY", df=data, data_version="qdb_2024Q1")

2. 快速加载（<10ms目标）
   df = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-02")

3. RL训练采样
   batch = qdb.sample(symbol="AAPL", window=1000)

4. 版本管理
   versions = qdb.list_versions(symbol="SPY", experiment_id="RL_v3")

性能目标：
- 加载延迟: < 10ms
- 支持并行: 多策略同时训练
- 缓存命中率: > 80%
- 存储格式: Parquet (列式存储，自动压缩)
"""


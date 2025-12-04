"""
Quantitative Database (QDB) Module

核心目标：
1. 一致性（Consistency）- 实盘、回测、模拟都从同一数据源取数据
2. 可复现性（Reproducibility）- 任何训练/实验都能复现
3. 可扩展性（Scalability）- 数据量大时仍能快速读写、支持多策略并行

四个关键点：
1. 数据标准化（Data Schema）
2. 数据索引与快速检索（Indexing）
3. 数据缓存与共享（Caching & Memory Mapping）
4. 数据版本与实验追踪（Versioning）
"""

from .qdb import QDB, create_qdb
from .schema import MarketDataSchema, validate_schema, normalize_dataframe
from .indexer import DataIndexer, IndexMetadata
from .cache import DataCache, CacheConfig
from .versioning import DataVersioning, VersionMetadata
from .ingestion import RealtimeCollector, HistoricalDownloader, create_realtime_collector, create_historical_downloader

__all__ = [
    'QDB',
    'create_qdb',
    'MarketDataSchema',
    'validate_schema',
    'normalize_dataframe',
    'DataIndexer',
    'IndexMetadata',
    'DataCache',
    'CacheConfig',
    'DataVersioning',
    'VersionMetadata',
    'RealtimeCollector',
    'HistoricalDownloader',
    'create_realtime_collector',
    'create_historical_downloader',
]

__version__ = '1.0.0'


"""
QDB主模块 - Quantitative Database

整合所有功能：
1. 数据标准化（Schema）
2. 数据索引与快速检索（Indexer）
3. 数据缓存与共享（Cache）
4. 数据版本与实验追踪（Versioning）

使用示例：
    # 初始化QDB
    qdb = QDB(base_path="./Data/datasets/qdb", memmap=True)
    
    # 存储数据
    qdb.store(symbol="SPY", df=market_data, data_version="qdb_2024Q1")
    
    # 快速加载（<10ms目标）
    df = qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-02")
    
    # RL训练采样
    batch = qdb.sample(symbol="AAPL", window=1000)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, date
from pathlib import Path
import logging

from .schema import MarketDataSchema, normalize_dataframe, validate_schema
from .indexer import DataIndexer, IndexMetadata
from .cache import DataCache, CacheConfig
from .versioning import DataVersioning, VersionMetadata

logger = logging.getLogger(__name__)


class QDB:
    """
    Quantitative Database - 量化数据库核心类
    
    核心目标：
    1. 一致性（Consistency）- 实盘、回测、模拟都从同一数据源取数据
    2. 可复现性（Reproducibility）- 任何训练/实验都能复现
    3. 可扩展性（Scalability）- 数据量大时仍能快速读写、支持多策略并行
    """
    
    def __init__(self,
                 base_path: str = "./Data/datasets/qdb",
                 memmap: bool = True,
                 cache_config: CacheConfig = None):
        """
        初始化QDB
        
        Args:
            base_path: 数据存储基础路径
            memmap: 是否启用内存映射（用于多进程共享）
            cache_config: 缓存配置
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.memmap = memmap
        
        # 初始化各个组件
        self.indexer = DataIndexer(base_path=str(self.base_path))
        self.cache = DataCache(config=cache_config or CacheConfig())
        self.versioning = DataVersioning(base_path=str(self.base_path))
        
        logger.info(f"QDB initialized at {self.base_path}")
    
    def store(self,
              symbol: str,
              df: pd.DataFrame,
              data_version: str = "1.0",
              source_format: str = "standard",
              experiment_id: str = "",
              feature_version: str = "default",
              description: str = "",
              tags: List[str] = None) -> str:
        """
        存储数据到QDB
        
        Args:
            symbol: 交易标的
            df: 数据DataFrame
            data_version: 数据版本（如 qdb_2024Q1）
            source_format: 数据源格式（用于标准化）
            experiment_id: 实验ID（如 RL_v3）
            feature_version: 特征版本（如 features_v7）
            description: 描述
            tags: 标签列表
        
        Returns:
            存储的文件路径
        """
        # 1. 数据标准化
        df_normalized = normalize_dataframe(df, source_format=source_format, symbol=symbol)
        
        # 2. Schema验证
        is_valid, errors = validate_schema(df_normalized)
        if not is_valid:
            raise ValueError(f"Schema validation failed: {errors}")
        
        # 3. 存储到索引器
        file_path = self.indexer.add_data(
            symbol=symbol,
            df=df_normalized,
            data_version=data_version
        )
        
        # 4. 更新版本信息
        version_id = self.versioning.create_version(
            symbol=symbol,
            data_version=data_version,
            feature_version=feature_version,
            experiment_id=experiment_id,
            description=description,
            tags=tags,
            file_path=self.base_path / file_path
        )
        
        # 5. 更新缓存
        if len(df_normalized) > 0:
            start_time = df_normalized.index.min()
            end_time = df_normalized.index.max()
            self.cache.put(symbol, start_time, end_time, df_normalized)
        
        logger.info(f"Stored {len(df_normalized)} records for {symbol} (version: {version_id})")
        return file_path
    
    def load(self,
             symbol: str,
             start: Optional[str] = None,
             end: Optional[str] = None,
             use_cache: bool = True) -> pd.DataFrame:
        """
        快速加载数据（目标延迟 < 10ms）
        
        Args:
            symbol: 交易标的
            start: 开始时间（字符串，如 "2024-01-01"）
            end: 结束时间（字符串）
            use_cache: 是否使用缓存
        
        Returns:
            DataFrame
        """
        # 转换时间字符串
        start_time = None
        end_time = None
        
        if start:
            if isinstance(start, str):
                start_time = pd.to_datetime(start)
            else:
                start_time = start
        
        if end:
            if isinstance(end, str):
                end_time = pd.to_datetime(end)
            else:
                end_time = end
        
        # 1. 尝试从缓存获取
        if use_cache and start_time and end_time:
            cached_df = self.cache.get(symbol, start_time, end_time)
            if cached_df is not None:
                logger.debug(f"Cache hit for {symbol}")
                return cached_df
        
        # 2. 从索引器加载
        df = self.indexer.load(symbol, start_time, end_time)
        
        if len(df) == 0:
            logger.warning(f"No data found for {symbol} in range {start_time} to {end_time}")
            return pd.DataFrame()
        
        # 3. 更新缓存
        if use_cache and len(df) > 0:
            actual_start = df.index.min()
            actual_end = df.index.max()
            self.cache.put(symbol, actual_start, actual_end, df)
        
        return df
    
    def sample(self,
               symbol: str,
               window: int = 1000,
               start: Optional[str] = None,
               end: Optional[str] = None) -> pd.DataFrame:
        """
        采样数据（用于RL训练）
        
        Args:
            symbol: 交易标的
            window: 采样窗口大小
            start: 开始时间（可选）
            end: 结束时间（可选）
        
        Returns:
            采样后的DataFrame
        """
        # 先尝试从缓存采样
        start_time = pd.to_datetime(start) if start else None
        end_time = pd.to_datetime(end) if end else None
        
        cached_sample = self.cache.sample(symbol, window, start_time, end_time)
        if cached_sample is not None:
            return cached_sample
        
        # 从索引器加载
        df = self.load(symbol, start, end, use_cache=True)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # 随机采样
        if len(df) > window:
            sampled = df.sample(n=window)
            return sampled.sort_index()
        
        return df
    
    def list_symbols(self) -> List[str]:
        """列出所有已存储的symbol"""
        return self.indexer.list_symbols()
    
    def get_time_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """获取symbol的时间范围"""
        return self.indexer.get_time_range(symbol)
    
    def get_metadata(self, symbol: str) -> Optional[IndexMetadata]:
        """获取symbol的索引元数据"""
        return self.indexer.get_metadata(symbol)
    
    def get_version(self, version_id: str) -> Optional[VersionMetadata]:
        """获取版本信息"""
        return self.versioning.get_version(version_id)
    
    def list_versions(self,
                     symbol: Optional[str] = None,
                     experiment_id: Optional[str] = None) -> List[VersionMetadata]:
        """列出版本"""
        return self.versioning.list_versions(symbol=symbol, experiment_id=experiment_id)
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def delete_symbol(self, symbol: str):
        """删除symbol的所有数据"""
        self.indexer.delete_symbol(symbol)
        self.cache.remove(symbol)
        logger.info(f"Deleted all data for {symbol}")
    
    def validate_data(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        验证数据完整性
        
        Args:
            symbol: 交易标的
            start: 开始时间
            end: 结束时间
        
        Returns:
            (is_valid, error_messages)
        """
        df = self.load(symbol, start, end)
        if len(df) == 0:
            return False, ["No data found"]
        
        return validate_schema(df)
    
    def get_info(self) -> Dict:
        """获取QDB信息"""
        symbols = self.list_symbols()
        
        return {
            'base_path': str(self.base_path),
            'memmap_enabled': self.memmap,
            'n_symbols': len(symbols),
            'symbols': symbols,
            'cache_stats': self.get_cache_stats(),
        }


# 便捷函数
def create_qdb(base_path: str = "./Data/datasets/qdb", memmap: bool = True) -> QDB:
    """创建QDB实例的便捷函数"""
    return QDB(base_path=base_path, memmap=memmap)
















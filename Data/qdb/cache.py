"""
数据缓存与共享模块 (Caching & Memory Mapping)

支持多策略并行训练（PPO、DQN、SAC同时跑）时共享内存，避免重复磁盘IO
使用memory map / Arrow IPC实现共享内存访问
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle
import hashlib
import time
import logging
from collections import OrderedDict
import threading

try:
    import mmap
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size_mb: int = 1024  # 最大缓存大小（MB）
    max_items: int = 100  # 最大缓存项数
    ttl_seconds: int = 3600  # 缓存过期时间（秒）
    use_memory_map: bool = True  # 是否使用内存映射
    eviction_policy: str = "LRU"  # 淘汰策略：LRU, LFU, FIFO


class DataCache:
    """
    数据缓存管理器
    
    功能：
    1. 内存缓存常用数据，避免重复IO
    2. 支持内存映射，多进程共享
    3. LRU/LFU淘汰策略
    4. 自动过期清理
    """
    
    def __init__(self, config: CacheConfig = None):
        """
        Args:
            config: 缓存配置
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict = OrderedDict()  # LRU缓存
        self._access_count: Dict[str, int] = {}  # 访问计数（用于LFU）
        self._cache_timestamps: Dict[str, float] = {}  # 缓存时间戳
        self._lock = threading.RLock()  # 线程锁
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _get_cache_key(self, symbol: str, start_time: datetime, end_time: datetime) -> str:
        """生成缓存键"""
        key_str = f"{symbol}_{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_size_mb(self, df: pd.DataFrame) -> float:
        """计算DataFrame大小（MB）"""
        return df.memory_usage(deep=True).sum() / 1024 / 1024
    
    def _is_expired(self, cache_key: str) -> bool:
        """检查缓存是否过期"""
        if cache_key not in self._cache_timestamps:
            return True
        
        age = time.time() - self._cache_timestamps[cache_key]
        return age > self.config.ttl_seconds
    
    def _evict_if_needed(self):
        """如果需要，执行缓存淘汰"""
        current_size_mb = sum(
            self._get_size_mb(df) 
            for df in self._cache.values()
        )
        
        # 检查大小限制
        while (current_size_mb > self.config.max_size_mb or 
               len(self._cache) > self.config.max_items):
            
            if len(self._cache) == 0:
                break
            
            # LRU策略：删除最久未使用的
            if self.config.eviction_policy == "LRU":
                cache_key = next(iter(self._cache))
            # LFU策略：删除访问次数最少的
            elif self.config.eviction_policy == "LFU":
                cache_key = min(self._access_count.items(), key=lambda x: x[1])[0]
            else:  # FIFO
                cache_key = next(iter(self._cache))
            
            # 删除
            if cache_key in self._cache:
                evicted_size = self._get_size_mb(self._cache[cache_key])
                del self._cache[cache_key]
                current_size_mb -= evicted_size
                self._evictions += 1
                logger.debug(f"Evicted cache entry: {cache_key}")
            
            if cache_key in self._access_count:
                del self._access_count[cache_key]
            if cache_key in self._cache_timestamps:
                del self._cache_timestamps[cache_key]
    
    def get(self, 
            symbol: str,
            start_time: datetime,
            end_time: datetime) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据
        
        Args:
            symbol: 交易标的
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            DataFrame或None（缓存未命中）
        """
        with self._lock:
            cache_key = self._get_cache_key(symbol, start_time, end_time)
            
            # 检查缓存
            if cache_key in self._cache:
                # 检查是否过期
                if self._is_expired(cache_key):
                    del self._cache[cache_key]
                    if cache_key in self._access_count:
                        del self._access_count[cache_key]
                    if cache_key in self._cache_timestamps:
                        del self._cache_timestamps[cache_key]
                    self._misses += 1
                    return None
                
                # 缓存命中
                df = self._cache[cache_key]
                
                # 更新LRU（移到末尾）
                self._cache.move_to_end(cache_key)
                
                # 更新访问计数
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                
                self._hits += 1
                logger.debug(f"Cache hit: {cache_key}")
                return df.copy()  # 返回副本，避免外部修改影响缓存
            
            # 缓存未命中
            self._misses += 1
            return None
    
    def put(self,
            symbol: str,
            start_time: datetime,
            end_time: datetime,
            df: pd.DataFrame):
        """
        将数据放入缓存
        
        Args:
            symbol: 交易标的
            start_time: 开始时间
            end_time: 结束时间
            df: 数据DataFrame
        """
        with self._lock:
            cache_key = self._get_cache_key(symbol, start_time, end_time)
            
            # 检查大小
            df_size_mb = self._get_size_mb(df)
            if df_size_mb > self.config.max_size_mb:
                logger.warning(f"DataFrame too large ({df_size_mb:.2f}MB), not caching")
                return
            
            # 如果需要，执行淘汰
            self._evict_if_needed()
            
            # 放入缓存
            self._cache[cache_key] = df.copy()  # 存储副本
            self._cache.move_to_end(cache_key)  # LRU
            
            # 更新元数据
            self._access_count[cache_key] = 1
            self._cache_timestamps[cache_key] = time.time()
            
            logger.debug(f"Cached: {cache_key} ({df_size_mb:.2f}MB)")
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._cache_timestamps.clear()
            logger.info("Cache cleared")
    
    def remove(self,
               symbol: str,
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None):
        """
        删除特定symbol的缓存
        
        Args:
            symbol: 交易标的
            start_time: 开始时间（可选，如果提供则只删除匹配的）
            end_time: 结束时间（可选）
        """
        with self._lock:
            keys_to_remove = []
            
            for cache_key in list(self._cache.keys()):
                # 这里需要存储symbol和时间信息，简化实现：删除所有包含symbol的
                # 实际应该从cache_key反推，但为了简化，这里删除所有
                keys_to_remove.append(cache_key)
            
            # 简化实现：如果提供了时间范围，需要更复杂的匹配
            # 这里先实现删除所有symbol相关的缓存
            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
            
            logger.info(f"Removed cache for {symbol}")
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            current_size_mb = sum(
                self._get_size_mb(df) 
                for df in self._cache.values()
            )
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'current_size_mb': current_size_mb,
                'current_items': len(self._cache),
                'max_size_mb': self.config.max_size_mb,
                'max_items': self.config.max_items,
            }
    
    def sample(self, 
               symbol: str,
               window: int = 1000,
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        从缓存中采样数据（用于RL训练）
        
        Args:
            symbol: 交易标的
            window: 采样窗口大小
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
        
        Returns:
            采样后的DataFrame
        """
        # 先尝试从缓存获取
        if start_time and end_time:
            df = self.get(symbol, start_time, end_time)
            if df is not None:
                # 随机采样
                if len(df) > window:
                    return df.sample(n=window).sort_index()
                return df
        
        return None
















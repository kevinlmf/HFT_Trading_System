"""
改进版优化索引器 - 修复了实现问题

改进点：
1. 修复二分查找的列表创建问题
2. 添加索引增量更新机制
3. Bloom Filter条件使用
4. 动态并行加载阈值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import bisect
import concurrent.futures
from collections import defaultdict
import logging

try:
    from pybloom_live import BloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False
    BloomFilter = None

logger = logging.getLogger(__name__)


class ImprovedOptimizedIndexer:
    """
    改进的优化索引器
    
    修复的问题：
    1. 二分查找时不再每次创建新列表
    2. 支持索引增量更新
    3. Bloom Filter条件使用
    4. 动态并行加载阈值
    """
    
    def __init__(self, base_path: str = "./Data/datasets/qdb", 
                 use_bloom_filter_threshold: int = 500):
        """
        Args:
            base_path: 数据存储路径
            use_bloom_filter_threshold: Symbol数量超过此阈值才使用Bloom Filter
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 优化的数据结构：
        # symbol -> (time_ranges_list, end_times_array) 元组
        # end_times_array预先计算，避免每次查询时创建
        self._time_index: Dict[str, Tuple[List, np.ndarray]] = {}
        
        # Bloom Filter：只在symbol数量大时使用
        self._use_bloom_filter = False
        self._bloom_filter: Optional[BloomFilter] = None
        self._bloom_threshold = use_bloom_filter_threshold
        
        # 元数据缓存
        self._metadata_cache: Dict[str, Dict] = {}
        
        # 加载索引
        self._build_optimized_index()
    
    def _build_optimized_index(self):
        """构建优化的索引结构"""
        index_file = self.base_path / "index.parquet"
        
        if not index_file.exists():
            return
        
        try:
            df = pd.read_parquet(index_file)
            symbols = df['symbol'].unique()
            
            # 决定是否使用Bloom Filter
            if BLOOM_AVAILABLE and len(symbols) > self._bloom_threshold:
                self._use_bloom_filter = True
                self._bloom_filter = BloomFilter(
                    capacity=len(symbols) * 2,  # 预留空间
                    error_rate=0.001
                )
            
            # 按symbol分组，构建排序数组
            for symbol in symbols:
                symbol_data = df[df['symbol'] == symbol]
                
                # 构建时间范围列表
                time_ranges = []
                for _, row in symbol_data.iterrows():
                    time_ranges.append((
                        pd.to_datetime(row['start_time']),
                        pd.to_datetime(row['end_time']),
                        row['file_path']
                    ))
                
                # 按start_time排序
                time_ranges.sort(key=lambda x: x[0])
                
                # 预先计算end_times数组（用于二分查找）
                end_times = np.array([tr[1].timestamp() for tr in time_ranges])
                
                # 存储元组：(time_ranges, end_times)
                self._time_index[symbol] = (time_ranges, end_times)
                
                # 添加到Bloom Filter（如果使用）
                if self._use_bloom_filter:
                    self._bloom_filter.add(symbol)
            
            logger.info(f"Built optimized index for {len(self._time_index)} symbols")
            if self._use_bloom_filter:
                logger.info(f"Bloom Filter enabled (threshold: {self._bloom_threshold})")
        except Exception as e:
            logger.error(f"Failed to build optimized index: {e}")
    
    def find_files_optimized(self,
                             symbol: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[str]:
        """
        优化的文件查找 - O(log n) 时间复杂度
        
        改进：使用预计算的end_times数组，避免每次创建新列表
        """
        # Bloom Filter快速检查（如果使用）
        if self._use_bloom_filter:
            if symbol not in self._bloom_filter:
                return []
        
        # 获取symbol的时间索引
        if symbol not in self._time_index:
            return []
        
        time_ranges, end_times = self._time_index[symbol]
        
        if len(time_ranges) == 0:
            return []
        
        # 如果没有时间限制，返回所有文件
        if start_time is None and end_time is None:
            return [fp for _, _, fp in time_ranges]
        
        # 使用numpy的searchsorted进行二分查找 - O(log n)
        # 查找第一个 end_time >= start_time 的位置
        if start_time is not None:
            start_ts = start_time.timestamp()
            # searchsorted返回插入位置，side='right'表示>=的位置
            idx = np.searchsorted(end_times, start_ts, side='right')
        else:
            idx = 0
        
        # 从idx开始，找到所有重叠的文件 - O(k)，k是重叠文件数
        matching_files = []
        for i in range(idx, len(time_ranges)):
            st, et, fp = time_ranges[i]
            
            # 检查是否重叠
            if start_time is not None and start_time > et:
                continue
            if end_time is not None and end_time < st:
                break  # 由于已排序，后续文件也不会重叠
            
            matching_files.append(fp)
        
        return matching_files
    
    def add_data(self,
                 symbol: str,
                 start_time: datetime,
                 end_time: datetime,
                 file_path: str):
        """
        增量添加数据，保持索引有序
        
        改进：支持增量更新，不需要重建整个索引
        """
        time_range = (start_time, end_time, file_path)
        
        if symbol not in self._time_index:
            # 新symbol，创建索引
            time_ranges = [time_range]
            end_times = np.array([end_time.timestamp()])
            self._time_index[symbol] = (time_ranges, end_times)
        else:
            # 现有symbol，插入到有序位置
            time_ranges, end_times = self._time_index[symbol]
            
            # 使用bisect找到插入位置（兼容Python < 3.10）
            start_ts = start_time.timestamp()
            insert_pos = bisect.bisect_left([tr[0].timestamp() for tr in time_ranges], start_ts)
            time_ranges.insert(insert_pos, time_range)
            
            # 更新end_times数组
            end_times = np.array([tr[1].timestamp() for tr in time_ranges])
            self._time_index[symbol] = (time_ranges, end_times)
        
        # 更新Bloom Filter（如果使用）
        if self._use_bloom_filter and symbol not in self._bloom_filter:
            self._bloom_filter.add(symbol)
    
    def load_parallel(self,
                     symbol: str,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     max_workers: int = 4,
                     parallel_threshold: int = 5) -> pd.DataFrame:
        """
        并行加载文件 - 带动态阈值
        
        改进：小文件数时串行加载，避免线程创建开销
        """
        file_paths = self.find_files_optimized(symbol, start_time, end_time)
        
        if len(file_paths) == 0:
            return pd.DataFrame()
        
        def load_file(rel_path: str) -> Optional[pd.DataFrame]:
            """加载单个文件"""
            try:
                full_path = self.base_path / rel_path
                return pd.read_parquet(full_path)
            except Exception as e:
                logger.error(f"Failed to load {rel_path}: {e}")
                return None
        
        # 根据文件数决定是否并行
        if len(file_paths) < parallel_threshold:
            # 串行加载（小文件数时更快）
            dfs = []
            for fp in file_paths:
                df = load_file(fp)
                if df is not None:
                    dfs.append(df)
        else:
            # 并行加载
            dfs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(load_file, fp): fp for fp in file_paths}
                
                for future in concurrent.futures.as_completed(futures):
                    df = future.result()
                    if df is not None:
                        dfs.append(df)
        
        if len(dfs) == 0:
            return pd.DataFrame()
        
        # 合并数据
        result = pd.concat(dfs, axis=0, ignore_index=False, sort=False)
        result = result.sort_index()
        
        # 去重
        if result.index.duplicated().any():
            result = result[~result.index.duplicated(keep='last')]
        
        # 时间范围过滤
        if start_time is not None:
            mask = result.index >= start_time
            result = result[mask]
        if end_time is not None:
            mask = result.index <= end_time
            result = result[mask]
        
        return result


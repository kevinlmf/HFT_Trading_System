"""
数据索引与快速检索模块 (Indexing)

实现快速时间范围查询：qdb.load(symbol="SPY", start="2024-01-01", end="2024-01-02")
延迟 < 10ms

使用Parquet + 索引表实现随机读取，无需全量加载
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """索引元数据"""
    symbol: str
    start_time: datetime
    end_time: datetime
    n_records: int
    file_path: str
    file_size_bytes: int
    created_at: datetime
    data_version: str = "1.0"
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'created_at': self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'IndexMetadata':
        d = d.copy()
        d['start_time'] = pd.to_datetime(d['start_time'])
        d['end_time'] = pd.to_datetime(d['end_time'])
        d['created_at'] = pd.to_datetime(d['created_at'])
        return cls(**d)


class DataIndexer:
    """
    数据索引器
    
    功能：
    1. 为每个symbol创建时间索引
    2. 支持快速时间范围查询
    3. 使用Parquet格式存储，支持列式存储和压缩
    4. 维护索引元数据，实现快速定位
    """
    
    def __init__(self, base_path: str = "./Data/datasets/qdb"):
        """
        Args:
            base_path: 数据存储基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.base_path / "index.parquet"
        self.metadata_file = self.base_path / "metadata.json"
        
        # 内存中的索引缓存
        self._index_cache: Optional[pd.DataFrame] = None
        self._metadata_cache: Dict[str, IndexMetadata] = {}
        
        # 加载现有索引
        self._load_index()
    
    def _load_index(self):
        """加载索引文件"""
        if self.index_file.exists():
            try:
                self._index_cache = pd.read_parquet(self.index_file)
                logger.info(f"Loaded index with {len(self._index_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index_cache = pd.DataFrame()
        else:
            self._index_cache = pd.DataFrame(columns=[
                'symbol', 'start_time', 'end_time', 'n_records', 
                'file_path', 'file_size_bytes', 'created_at', 'data_version'
            ])
        
        # 加载元数据
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    for symbol, meta_dict in metadata_dict.items():
                        self._metadata_cache[symbol] = IndexMetadata.from_dict(meta_dict)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
    
    def _save_index(self):
        """保存索引到文件"""
        if self._index_cache is not None and len(self._index_cache) > 0:
            self._index_cache.to_parquet(self.index_file, index=False)
            logger.debug(f"Saved index with {len(self._index_cache)} entries")
        
        # 保存元数据
        metadata_dict = {
            symbol: meta.to_dict() 
            for symbol, meta in self._metadata_cache.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def add_data(self, 
                 symbol: str,
                 df: pd.DataFrame,
                 data_version: str = "1.0") -> str:
        """
        添加数据到索引
        
        Args:
            symbol: 交易标的
            df: 数据DataFrame（必须有timestamp索引）
            data_version: 数据版本
        
        Returns:
            存储的文件路径
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # 确保时间排序
        df = df.sort_index()
        
        start_time = df.index.min()
        end_time = df.index.max()
        n_records = len(df)
        
        # 生成文件路径（按symbol和日期组织）
        symbol_dir = self.base_path / "data" / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名：symbol_YYYYMMDD_YYYYMMDD.parquet
        start_str = start_time.strftime("%Y%m%d")
        end_str = end_time.strftime("%Y%m%d")
        filename = f"{symbol}_{start_str}_{end_str}.parquet"
        file_path = symbol_dir / filename
        
        # 保存数据为Parquet（列式存储，自动压缩）
        df.to_parquet(file_path, compression='snappy', index=True)
        file_size = file_path.stat().st_size
        
        # 创建索引元数据
        metadata = IndexMetadata(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            n_records=n_records,
            file_path=str(file_path.relative_to(self.base_path)),
            file_size_bytes=file_size,
            created_at=datetime.now(),
            data_version=data_version,
        )
        
        # 更新索引缓存
        new_row = pd.DataFrame([{
            'symbol': symbol,
            'start_time': start_time,
            'end_time': end_time,
            'n_records': n_records,
            'file_path': str(file_path.relative_to(self.base_path)),
            'file_size_bytes': file_size,
            'created_at': datetime.now(),
            'data_version': data_version,
        }])
        
        if self._index_cache is None or len(self._index_cache) == 0:
            self._index_cache = new_row
        else:
            # 检查是否已存在（相同时间范围）
            existing = self._index_cache[
                (self._index_cache['symbol'] == symbol) &
                (self._index_cache['start_time'] == start_time) &
                (self._index_cache['end_time'] == end_time)
            ]
            if len(existing) > 0:
                # 更新现有记录
                idx = existing.index[0]
                self._index_cache.loc[idx] = new_row.iloc[0]
            else:
                # 添加新记录
                self._index_cache = pd.concat([self._index_cache, new_row], ignore_index=True)
        
        # 更新元数据缓存
        self._metadata_cache[symbol] = metadata
        
        # 保存索引
        self._save_index()
        
        logger.info(f"Added {n_records} records for {symbol} ({start_time} to {end_time})")
        return str(file_path)
    
    def find_files(self,
                   symbol: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[str]:
        """
        查找匹配时间范围的文件
        
        Args:
            symbol: 交易标的
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
        
        Returns:
            匹配的文件路径列表（相对路径）
        """
        if self._index_cache is None or len(self._index_cache) == 0:
            return []
        
        # 筛选symbol
        symbol_data = self._index_cache[self._index_cache['symbol'] == symbol]
        
        if len(symbol_data) == 0:
            return []
        
        # 筛选时间范围（文件的时间范围与查询范围有重叠）
        if start_time is not None:
            symbol_data = symbol_data[symbol_data['end_time'] >= start_time]
        
        if end_time is not None:
            symbol_data = symbol_data[symbol_data['start_time'] <= end_time]
        
        # 返回文件路径列表
        file_paths = symbol_data['file_path'].tolist()
        return file_paths
    
    def load(self,
             symbol: str,
             start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        快速加载数据（延迟 < 10ms目标）
        
        Args:
            symbol: 交易标的
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            合并后的DataFrame
        """
        import time
        start_load = time.time()
        
        # 查找匹配的文件
        file_paths = self.find_files(symbol, start_time, end_time)
        
        if len(file_paths) == 0:
            logger.warning(f"No data found for {symbol} in range {start_time} to {end_time}")
            return pd.DataFrame()
        
        # 加载所有匹配的文件
        dfs = []
        for rel_path in file_paths:
            full_path = self.base_path / rel_path
            try:
                df = pd.read_parquet(full_path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {full_path}: {e}")
                continue
        
        if len(dfs) == 0:
            return pd.DataFrame()
        
        # 合并数据
        result = pd.concat(dfs, axis=0)
        result = result.sort_index()
        
        # 去重（如果有重叠）
        result = result[~result.index.duplicated(keep='last')]
        
        # 时间范围过滤
        if start_time is not None:
            result = result[result.index >= start_time]
        if end_time is not None:
            result = result[result.index <= end_time]
        
        load_time = (time.time() - start_load) * 1000  # 转换为毫秒
        logger.debug(f"Loaded {len(result)} records for {symbol} in {load_time:.2f}ms")
        
        return result
    
    def get_metadata(self, symbol: str) -> Optional[IndexMetadata]:
        """获取symbol的元数据"""
        return self._metadata_cache.get(symbol)
    
    def list_symbols(self) -> List[str]:
        """列出所有已索引的symbol"""
        if self._index_cache is None or len(self._index_cache) == 0:
            return []
        return sorted(self._index_cache['symbol'].unique().tolist())
    
    def get_time_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """获取symbol的时间范围"""
        if self._index_cache is None or len(self._index_cache) == 0:
            return None
        
        symbol_data = self._index_cache[self._index_cache['symbol'] == symbol]
        if len(symbol_data) == 0:
            return None
        
        start_time = symbol_data['start_time'].min()
        end_time = symbol_data['end_time'].max()
        return (start_time, end_time)
    
    def delete_symbol(self, symbol: str):
        """删除symbol的所有数据"""
        if self._index_cache is None:
            return
        
        # 获取所有文件路径
        symbol_data = self._index_cache[self._index_cache['symbol'] == symbol]
        file_paths = symbol_data['file_path'].tolist()
        
        # 删除文件
        for rel_path in file_paths:
            full_path = self.base_path / rel_path
            if full_path.exists():
                full_path.unlink()
        
        # 删除目录（如果为空）
        symbol_dir = self.base_path / "data" / symbol
        if symbol_dir.exists():
            try:
                symbol_dir.rmdir()
            except OSError:
                pass  # 目录不为空，保留
        
        # 从索引中删除
        self._index_cache = self._index_cache[self._index_cache['symbol'] != symbol]
        
        # 从元数据中删除
        if symbol in self._metadata_cache:
            del self._metadata_cache[symbol]
        
        # 保存
        self._save_index()
        logger.info(f"Deleted all data for {symbol}")
















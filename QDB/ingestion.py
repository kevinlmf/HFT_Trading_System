"""
数据摄取层 (Data Ingestion Layer)

将不同数据源的数据导入QDB：
- 实时数据收集（realtime_collector.py）
- 历史数据下载（historical_downloader.py）
"""

import asyncio
import pandas as pd
from typing import List, Optional, Callable, Dict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from Market_Data.base_connector import BaseConnector, MarketTick
from QDB.qdb import QDB

logger = logging.getLogger(__name__)


class RealtimeCollector:
    """
    实时数据收集器
    
    从实时数据源（WebSocket/REST API）收集数据并存储到QDB
    """
    
    def __init__(self, 
                 connector: BaseConnector,
                 qdb: QDB,
                 buffer_size: int = 1000):
        """
        Args:
            connector: 数据连接器（如AlpacaConnector）
            qdb: QDB实例
            buffer_size: 缓冲区大小（达到此大小后批量写入）
        """
        self.connector = connector
        self.qdb = qdb
        self.buffer_size = buffer_size
        
        # 数据缓冲区：{symbol: [ticks]}
        self.buffers: Dict[str, List[MarketTick]] = {}
        
        # 注册回调
        self.connector.on_tick(self._on_tick)
    
    def _on_tick(self, tick: MarketTick):
        """处理实时tick数据"""
        symbol = tick.symbol
        
        if symbol not in self.buffers:
            self.buffers[symbol] = []
        
        self.buffers[symbol].append(tick)
        
        # 达到缓冲区大小，批量写入
        if len(self.buffers[symbol]) >= self.buffer_size:
            self._flush_buffer(symbol)
    
    def _flush_buffer(self, symbol: str):
        """将缓冲区数据写入QDB"""
        if symbol not in self.buffers or len(self.buffers[symbol]) == 0:
            return
        
        ticks = self.buffers[symbol]
        
        # 转换为DataFrame
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'bid_price': tick.bid,
                'ask_price': tick.ask,
                'bid_size': tick.bid_size,
                'ask_size': tick.ask_size,
                'last_price': tick.price,
                'volume': tick.volume,
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # 存储到QDB
        try:
            self.qdb.store(
                symbol=symbol,
                df=df,
                data_version=f"realtime_{datetime.now().strftime('%Y%m%d')}",
                source_format="standard"
            )
            logger.info(f"Flushed {len(ticks)} ticks for {symbol}")
        except Exception as e:
            logger.error(f"Failed to flush buffer for {symbol}: {e}")
        
        # 清空缓冲区
        self.buffers[symbol] = []
    
    async def start(self, symbols: List[str]):
        """启动实时收集"""
        await self.connector.start()
        await self.connector.subscribe(symbols)
        logger.info(f"Started realtime collection for {symbols}")
    
    async def stop(self):
        """停止收集并刷新所有缓冲区"""
        # 刷新所有缓冲区
        for symbol in list(self.buffers.keys()):
            self._flush_buffer(symbol)
        
        await self.connector.stop()
        logger.info("Stopped realtime collection")
    
    def flush_all(self):
        """手动刷新所有缓冲区"""
        for symbol in list(self.buffers.keys()):
            self._flush_buffer(symbol)


class HistoricalDownloader:
    """
    历史数据下载器
    
    从数据源下载历史数据并存储到QDB
    """
    
    def __init__(self, qdb: QDB):
        """
        Args:
            qdb: QDB实例
        """
        self.qdb = qdb
    
    def download_from_dataframe(self,
                               symbol: str,
                               df: pd.DataFrame,
                               data_version: str = "historical",
                               source_format: str = "standard",
                               experiment_id: str = "",
                               description: str = "") -> str:
        """
        从DataFrame下载数据到QDB
        
        Args:
            symbol: 交易标的
            df: 历史数据DataFrame
            data_version: 数据版本
            source_format: 数据源格式
            experiment_id: 实验ID
            description: 描述
        
        Returns:
            存储的文件路径
        """
        return self.qdb.store(
            symbol=symbol,
            df=df,
            data_version=data_version,
            source_format=source_format,
            experiment_id=experiment_id,
            description=description
        )
    
    def download_from_file(self,
                          symbol: str,
                          file_path: str,
                          data_version: str = "historical",
                          source_format: str = "standard",
                          **kwargs) -> str:
        """
        从文件下载数据到QDB
        
        Args:
            symbol: 交易标的
            file_path: 文件路径（CSV/Parquet）
            data_version: 数据版本
            source_format: 数据源格式
            **kwargs: 其他参数传递给qdb.store
        
        Returns:
            存储的文件路径
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 读取文件
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self.qdb.store(
            symbol=symbol,
            df=df,
            data_version=data_version,
            source_format=source_format,
            **kwargs
        )
    
    def download_batch(self,
                      symbols: List[str],
                      download_func: Callable[[str], pd.DataFrame],
                      data_version: str = "historical",
                      **kwargs) -> Dict[str, str]:
        """
        批量下载多个symbol的数据
        
        Args:
            symbols: 交易标的列表
            download_func: 下载函数，接受symbol，返回DataFrame
            data_version: 数据版本
            **kwargs: 其他参数
        
        Returns:
            {symbol: file_path} 字典
        """
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Downloading {symbol}...")
                df = download_func(symbol)
                
                file_path = self.qdb.store(
                    symbol=symbol,
                    df=df,
                    data_version=data_version,
                    **kwargs
                )
                
                results[symbol] = file_path
                logger.info(f"Downloaded {symbol}: {len(df)} records")
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                results[symbol] = None
        
        return results


# 便捷函数
def create_realtime_collector(connector: BaseConnector, 
                              qdb: QDB,
                              buffer_size: int = 1000) -> RealtimeCollector:
    """创建实时收集器"""
    return RealtimeCollector(connector, qdb, buffer_size)


def create_historical_downloader(qdb: QDB) -> HistoricalDownloader:
    """创建历史数据下载器"""
    return HistoricalDownloader(qdb)


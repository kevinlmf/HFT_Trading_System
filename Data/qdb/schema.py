"""
数据标准化模块 (Data Schema)

确保所有数据源（Binance / Polygon / Yahoo）都统一为标准格式：
timestamp, bid_price, bid_size, ask_price, ask_size, last_price, volume
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketDataSchema:
    """
    标准化的市场数据Schema
    
    所有数据源都必须转换为这个格式，确保一致性
    """
    # 必需字段
    timestamp: pd.DatetimeIndex
    symbol: str
    
    # 价格字段
    bid_price: Optional[pd.Series] = None
    bid_size: Optional[pd.Series] = None
    ask_price: Optional[pd.Series] = None
    ask_size: Optional[pd.Series] = None
    last_price: Optional[pd.Series] = None
    
    # 交易量
    volume: Optional[pd.Series] = None
    
    # 可选字段（OHLC）
    open: Optional[pd.Series] = None
    high: Optional[pd.Series] = None
    low: Optional[pd.Series] = None
    close: Optional[pd.Series] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
        }
        
        # 添加可选字段
        for field in ['bid_price', 'bid_size', 'ask_price', 'ask_size', 
                     'last_price', 'volume', 'open', 'high', 'low', 'close']:
            value = getattr(self, field)
            if value is not None:
                data[field] = value
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, symbol: str) -> 'MarketDataSchema':
        """从DataFrame创建Schema"""
        # 确保timestamp是索引
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return cls(
            timestamp=df.index,
            symbol=symbol,
            bid_price=df.get('bid_price') or df.get('bid'),
            bid_size=df.get('bid_size') or df.get('bidSize'),
            ask_price=df.get('ask_price') or df.get('ask'),
            ask_size=df.get('ask_size') or df.get('askSize'),
            last_price=df.get('last_price') or df.get('price') or df.get('close'),
            volume=df.get('volume') or df.get('Volume'),
            open=df.get('open') or df.get('Open'),
            high=df.get('high') or df.get('High'),
            low=df.get('low') or df.get('Low'),
            close=df.get('close') or df.get('Close'),
        )


# 标准字段映射表（用于不同数据源的适配）
FIELD_MAPPINGS = {
    # Alpaca格式
    'alpaca': {
        'timestamp': 't',
        'price': 'p',
        'volume': 's',
        'bid': 'bp',
        'ask': 'ap',
        'bid_size': 'bs',
        'ask_size': 'as',
    },
    # Polygon格式
    'polygon': {
        'timestamp': 'timestamp',
        'price': 'last_price',
        'volume': 'volume',
        'bid': 'bid_price',
        'ask': 'ask_price',
        'bid_size': 'bid_size',
        'ask_size': 'ask_size',
    },
    # Yahoo Finance格式
    'yahoo': {
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    },
    # Binance格式
    'binance': {
        'timestamp': 'timestamp',
        'price': 'price',
        'volume': 'volume',
        'bid': 'best_bid_price',
        'ask': 'best_ask_price',
        'bid_size': 'best_bid_qty',
        'ask_size': 'best_ask_qty',
    },
}


def normalize_dataframe(df: pd.DataFrame, 
                       source_format: str = 'standard',
                       symbol: str = None) -> pd.DataFrame:
    """
    将不同来源的数据标准化为标准格式
    
    Args:
        df: 原始DataFrame
        source_format: 数据源格式 ('alpaca', 'polygon', 'yahoo', 'binance', 'standard')
        symbol: 交易标的符号
    
    Returns:
        标准化后的DataFrame，包含标准字段：
        timestamp, symbol, bid_price, bid_size, ask_price, ask_size, 
        last_price, volume, open, high, low, close
    """
    df = df.copy()
    
    # 如果已有标准格式，直接返回
    if source_format == 'standard':
        if 'timestamp' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have timestamp index or column")
        return df
    
    # 获取字段映射
    mapping = FIELD_MAPPINGS.get(source_format.lower())
    if not mapping:
        raise ValueError(f"Unknown source format: {source_format}")
    
    # 标准化timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # 尝试从列名推断
        for ts_col in ['t', 'time', 'date', 'Date', 'datetime']:
            if ts_col in df.columns:
                df['timestamp'] = pd.to_datetime(df[ts_col])
                df.set_index('timestamp', inplace=True)
                break
        else:
            raise ValueError("Cannot find timestamp column")
    
    # 重命名列
    rename_map = {}
    for std_field, src_field in mapping.items():
        if src_field in df.columns:
            rename_map[src_field] = std_field
    
    df.rename(columns=rename_map, inplace=True)
    
    # 添加symbol列
    if symbol:
        df['symbol'] = symbol
    elif 'symbol' not in df.columns:
        # 尝试从索引或列名推断
        if hasattr(df, 'name') and df.name:
            df['symbol'] = df.name
        else:
            df['symbol'] = 'UNKNOWN'
    
    # 确保标准字段存在（用None填充缺失字段）
    standard_fields = ['bid_price', 'bid_size', 'ask_price', 'ask_size', 
                      'last_price', 'volume', 'open', 'high', 'low', 'close']
    
    for field in standard_fields:
        if field not in df.columns:
            # 尝试从其他字段推断
            if field == 'last_price' and 'close' in df.columns:
                df['last_price'] = df['close']
            elif field == 'last_price' and 'price' in df.columns:
                df['last_price'] = df['price']
            else:
                df[field] = np.nan
    
    # 确保数据类型正确
    price_fields = ['bid_price', 'ask_price', 'last_price', 'open', 'high', 'low', 'close']
    for field in price_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    size_fields = ['bid_size', 'ask_size', 'volume']
    for field in size_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
    
    return df


def validate_schema(df: pd.DataFrame, 
                   required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """
    验证DataFrame是否符合标准Schema
    
    Args:
        df: 要验证的DataFrame
        required_fields: 必需字段列表，默认['timestamp', 'symbol', 'last_price']
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if required_fields is None:
        required_fields = ['timestamp', 'symbol', 'last_price']
    
    # 检查timestamp索引
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")
    
    # 检查必需字段
    for field in required_fields:
        if field == 'timestamp':
            continue  # 已检查索引
        if field not in df.columns:
            errors.append(f"Missing required field: {field}")
    
    # 检查数据类型
    price_fields = ['bid_price', 'ask_price', 'last_price', 'open', 'high', 'low', 'close']
    for field in price_fields:
        if field in df.columns:
            if not pd.api.types.is_numeric_dtype(df[field]):
                errors.append(f"Field {field} must be numeric")
    
    # 检查时间顺序
    if len(df) > 1:
        if not df.index.is_monotonic_increasing:
            errors.append("Timestamp index must be monotonic increasing")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def create_empty_schema(symbol: str, 
                       start_time: datetime, 
                       end_time: datetime,
                       freq: str = '1min') -> MarketDataSchema:
    """
    创建空的标准Schema（用于预分配）
    
    Args:
        symbol: 交易标的
        start_time: 开始时间
        end_time: 结束时间
        freq: 时间频率（pandas频率字符串）
    
    Returns:
        MarketDataSchema对象
    """
    timestamp = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    return MarketDataSchema(
        timestamp=timestamp,
        symbol=symbol,
        bid_price=pd.Series(index=timestamp, dtype=float),
        bid_size=pd.Series(index=timestamp, dtype=float),
        ask_price=pd.Series(index=timestamp, dtype=float),
        ask_size=pd.Series(index=timestamp, dtype=float),
        last_price=pd.Series(index=timestamp, dtype=float),
        volume=pd.Series(index=timestamp, dtype=float),
    )
















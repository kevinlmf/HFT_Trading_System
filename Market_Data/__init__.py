"""Market Data Connectors"""

from .base_connector import BaseConnector, MarketTick, OrderBookSnapshot
from .alpaca_connector import AlpacaConnector

# 可选连接器（需要安装相应依赖）
try:
    from .yahoo_connector import YahooFinanceConnector
except ImportError:
    YahooFinanceConnector = None

try:
    from .binance_connector import BinanceConnector
except ImportError:
    BinanceConnector = None

try:
    from .polygon_connector import PolygonConnector
except ImportError:
    PolygonConnector = None

try:
    from .alphavantage_connector import AlphaVantageConnector
except ImportError:
    AlphaVantageConnector = None

try:
    from .coinbase_connector import CoinbaseProConnector
except ImportError:
    CoinbaseProConnector = None

try:
    from .iexcloud_connector import IEXCloudConnector
except ImportError:
    IEXCloudConnector = None

__all__ = [
    'BaseConnector',
    'MarketTick',
    'OrderBookSnapshot',
    'AlpacaConnector',
]

# 添加可选连接器
if YahooFinanceConnector:
    __all__.append('YahooFinanceConnector')
if BinanceConnector:
    __all__.append('BinanceConnector')
if PolygonConnector:
    __all__.append('PolygonConnector')
if AlphaVantageConnector:
    __all__.append('AlphaVantageConnector')
if CoinbaseProConnector:
    __all__.append('CoinbaseProConnector')
if IEXCloudConnector:
    __all__.append('IEXCloudConnector')

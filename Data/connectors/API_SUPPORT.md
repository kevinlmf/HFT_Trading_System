# 支持的API和数据源

本交易系统设计为可扩展的架构，可以轻松接入各种市场数据API和交易平台。

## 当前已支持的API

### 🚀 HFT级连接器（适合高频交易）

#### 1. Alpaca Markets ✅ **推荐用于HFT**
- **类型**: 股票市场数据 + 交易执行
- **延迟**: ⚡ 实时（毫秒级）
- **免费层级**: ✅ 有（Paper Trading）
- **数据**: 实时股票报价、历史数据、订单簿
- **交易**: 支持Paper和Live交易
- **HFT适用性**: ⭐⭐⭐⭐⭐ 专业级，低延迟
- **实现**: `Data/connectors/alpaca_connector.py`

#### 2. Polygon.io ✅ **推荐用于HFT**
- **类型**: 股票、期权、外汇、加密货币
- **延迟**: ⚡ 实时（毫秒级）
- **免费层级**: ✅ 有（基础数据）
- **数据**: WebSocket实时数据、历史数据API、订单簿深度
- **HFT适用性**: ⭐⭐⭐⭐⭐ 专业级，机构级数据
- **实现**: `Data/connectors/polygon_connector.py`

#### 3. Binance ✅ **加密货币HFT**
- **类型**: 加密货币现货和期货
- **延迟**: ⚡ 实时（毫秒级）
- **免费层级**: ✅ 有（Testnet）
- **数据**: WebSocket实时数据、深度数据、订单流
- **HFT适用性**: ⭐⭐⭐⭐ 加密货币HFT首选
- **实现**: `Data/connectors/binance_connector.py`

#### 4. Coinbase Pro ✅ **加密货币HFT**
- **类型**: 加密货币现货和期货
- **延迟**: ⚡ 实时（毫秒级）
- **免费层级**: ✅ 有（Sandbox用于测试，生产环境专业级）
- **数据**: WebSocket实时数据、订单簿深度、交易执行
- **HFT适用性**: ⭐⭐⭐⭐ 专业加密货币交易所，适合HFT
- **实现**: `Data/connectors/coinbase_connector.py`
- **注意**: 沙盒环境用于测试，生产环境提供专业级低延迟数据

---

### 🧪 测试/开发连接器（不适合HFT）

#### 5. Yahoo Finance ⚠️ **仅用于测试**
- **类型**: 股票、ETF、指数
- **延迟**: ⚠️ **15-20分钟延迟** - 不适合HFT
- **免费层级**: ✅ 完全免费
- **数据**: 历史数据丰富，实时报价（延迟）
- **特点**: 无需API密钥
- **HFT适用性**: ❌ **不适合HFT** - 仅用于策略开发和测试
- **实现**: `Data/connectors/yahoo_connector.py`

#### 6. Alpha Vantage ⚠️ **仅用于测试**
- **类型**: 股票、外汇、加密货币
- **延迟**: ⚠️ REST API，1分钟更新 - 不适合HFT
- **免费层级**: ✅ 有（500 calls/day）
- **数据**: REST API，技术指标数据，基本面数据
- **HFT适用性**: ❌ **不适合HFT** - REST API延迟高
- **实现**: `Data/connectors/alphavantage_connector.py`

#### 7. IEX Cloud ⚠️ **中等延迟**
- **类型**: 股票市场数据
- **延迟**: ⚡ 实时（但免费层级有限制）
- **免费层级**: ✅ 有（50,000 messages/month）
- **数据**: WebSocket实时数据、历史数据、基本面数据
- **HFT适用性**: ⚠️ 免费层级可能不够，付费版适合HFT
- **实现**: `Data/connectors/iexcloud_connector.py`

## 可以接入的其他API

### 股票市场

#### 1. Polygon.io
- **类型**: 股票、期权、外汇、加密货币
- **免费层级**: ✅ 有（基础数据）
- **特点**: 
  - WebSocket实时数据
  - 历史数据API
  - 期权链数据
- **文档**: https://polygon.io/docs

#### 2. Alpha Vantage
- **类型**: 股票、外汇、加密货币
- **免费层级**: ✅ 有（500 calls/day）
- **特点**: 
  - REST API
  - 技术指标数据
  - 基本面数据
- **文档**: https://www.alphavantage.co/documentation/

#### 3. Yahoo Finance (yfinance) ✅ 已实现
- **类型**: 股票、ETF、指数
- **免费层级**: ✅ 完全免费
- **特点**: 
  - 无需API密钥
  - 历史数据丰富
  - 实时报价（有延迟）
- **文档**: https://github.com/ranaroussi/yfinance

#### 4. IEX Cloud ✅ 已实现
- **类型**: 股票市场数据
- **免费层级**: ✅ 有（50,000 messages/month）
- **特点**: 
  - WebSocket实时数据
  - 历史数据
  - 基本面数据
- **文档**: https://iexcloud.io/docs/api/

#### 5. Finnhub
- **类型**: 股票、加密货币、外汇
- **免费层级**: ✅ 有（60 calls/minute）
- **特点**: 
  - WebSocket实时数据
  - 新闻数据
  - 技术指标
- **文档**: https://finnhub.io/docs/api

### 加密货币

#### 1. Binance ✅ 已实现
- **类型**: 加密货币现货和期货
- **免费层级**: ✅ 有
- **特点**: 
  - WebSocket实时数据
  - 深度数据
  - 交易执行
- **文档**: https://binance-docs.github.io/apidocs/

#### 2. Coinbase Pro ✅ 已实现
- **类型**: 加密货币
- **免费层级**: ✅ 有
- **特点**: 
  - WebSocket实时数据
  - 交易执行
- **文档**: https://docs.pro.coinbase.com/

#### 3. Kraken
- **类型**: 加密货币
- **免费层级**: ✅ 有
- **特点**: 
  - WebSocket实时数据
  - 深度数据
- **文档**: https://docs.kraken.com/websockets/

### 外汇

#### 1. OANDA
- **类型**: 外汇、CFD
- **免费层级**: ✅ 有（Practice账户）
- **特点**: 
  - REST API
  - 实时汇率
  - 历史数据
- **文档**: https://developer.oanda.com/

#### 2. FXCM
- **类型**: 外汇、CFD
- **免费层级**: ✅ 有（Demo账户）
- **特点**: 
  - REST API
  - WebSocket数据
- **文档**: https://www.fxcm.com/developers/

### 期货

#### 1. Interactive Brokers (IBKR)
- **类型**: 股票、期货、期权、外汇
- **免费层级**: ❌ 需要账户
- **特点**: 
  - TWS API
  - 专业级数据
  - 全球市场
- **文档**: https://interactivebrokers.github.io/tws-api/

#### 2. CME Group DataMine
- **类型**: 期货市场数据
- **免费层级**: ❌ 需要订阅
- **特点**: 
  - 历史数据
  - 实时数据（需订阅）
- **文档**: https://www.cmegroup.com/market-data/

## 如何添加新的API连接器

### 步骤1: 创建连接器类

继承 `BaseConnector` 并实现必要方法：

```python
# Data/connectors/your_api_connector.py
from .base_connector import BaseConnector, MarketTick
import asyncio

class YourAPIConnector(BaseConnector):
    def __init__(self, api_key: str, api_secret: str = None):
        super().__init__(api_key, api_secret)
        # 初始化你的API客户端
    
    async def connect(self):
        """建立连接"""
        # 实现连接逻辑
        self.is_connected = True
    
    async def disconnect(self):
        """断开连接"""
        # 实现断开逻辑
        self.is_connected = False
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的"""
        self.subscribed_symbols = symbols
        # 实现订阅逻辑
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        # 实现取消订阅逻辑
```

### 步骤2: 注册连接器

在 `Data/connectors/__init__.py` 中添加：

```python
from .your_api_connector import YourAPIConnector

__all__ = [
    'BaseConnector',
    'MarketTick',
    'AlpacaConnector',
    'YourAPIConnector',  # 添加这里
]
```

### 步骤3: 在交易引擎中使用

```python
from Data.connectors import YourAPIConnector

connector = YourAPIConnector(
    api_key="your_key",
    api_secret="your_secret"
)

engine = RealTimeTradingEngine(
    connector=connector,
    strategies=strategies,
    initial_capital=100000
)
```

## 推荐接入顺序

1. **Yahoo Finance** - 最简单，无需API密钥，适合测试
2. **Binance** - 加密货币，WebSocket完善，文档清晰
3. **Polygon.io** - 股票数据全面，免费层级友好
4. **IEX Cloud** - 股票数据，WebSocket支持好

## 注意事项

- **API限制**: 注意免费层级的调用限制
- **数据延迟**: 免费层级通常有15分钟延迟
- **认证方式**: 不同API的认证方式不同（API Key, OAuth等）
- **数据格式**: 需要将不同API的数据格式统一转换为 `MarketTick`
- **错误处理**: 实现重连机制和错误处理

## 示例：快速添加Yahoo Finance连接器

见 `Data/connectors/yahoo_connector.py` 示例实现。







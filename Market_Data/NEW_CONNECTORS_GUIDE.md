# 新增API连接器使用指南

## 新增的连接器

### 1. Alpha Vantage (股票数据)

**特点：**
- ✅ 免费：500 calls/day
- ✅ REST API，简单易用
- ✅ 提供技术指标和基本面数据
- ⚠️ 需要API密钥（免费注册）

**使用方法：**

```bash
# 1. 获取免费API密钥
# 访问：https://www.alphavantage.co/support/#api-key

# 2. 设置环境变量
export ALPHAVANTAGE_API_KEY='your_api_key_here'

# 3. 运行交易系统
./run_trading.sh paper --connector alphavantage --symbols AAPL,MSFT --interval 60
```

**注意事项：**
- 默认更新间隔为60秒（避免超过免费限制）
- 每天最多500次API调用
- 数据更新频率：1分钟

---

### 2. Coinbase Pro (加密货币)

**特点：**
- ✅ 免费沙盒环境
- ✅ WebSocket实时数据
- ✅ 无需API密钥（仅数据订阅）
- ✅ 支持BTC-USD, ETH-USD等格式

**使用方法：**

```bash
# 直接使用，无需API密钥
./run_trading.sh paper --connector coinbase --symbols BTC-USD,ETH-USD --interval 10
```

**注意事项：**
- 使用沙盒环境（sandbox=True）
- 标的格式：BTC-USD, ETH-USD（不是BTCUSDT）
- 需要安装：`pip install websockets`

---

### 3. IEX Cloud (股票数据)

**特点：**
- ✅ 免费：50,000 messages/month
- ✅ WebSocket实时数据
- ✅ 提供历史数据和基本面数据
- ⚠️ 需要API密钥（免费注册）

**使用方法：**

```bash
# 1. 获取免费API密钥
# 访问：https://iexcloud.io/

# 2. 设置环境变量
export IEXCLOUD_API_KEY='your_api_key_here'

# 3. 运行交易系统
./run_trading.sh paper --connector iexcloud --symbols AAPL,MSFT --interval 10
```

**注意事项：**
- 使用沙盒环境（sandbox=True）
- 每月最多50,000条消息
- WebSocket实时数据流

---

## 所有可用连接器对比

| 连接器 | 免费 | API密钥 | 延迟 | 最佳用途 |
|--------|------|---------|------|----------|
| **Yahoo Finance** | ✅ | ❌ | 15-20分钟 | 测试、开发 |
| **Alpaca** | ✅ | ✅ | 实时 | Paper交易 |
| **Binance** | ✅ | ❌ | 实时 | 加密货币 |
| **Polygon.io** | ✅ | ✅ | 实时 | 专业交易 |
| **Alpha Vantage** | ✅ | ✅ | 1分钟 | 股票数据（500次/天） |
| **Coinbase Pro** | ✅ | ❌ | 实时 | 加密货币（沙盒） |
| **IEX Cloud** | ✅ | ✅ | 实时 | 股票数据（50k消息/月） |

---

## 快速开始（无需API密钥）

如果你想立即开始测试，推荐使用：

```bash
# 选项1：Yahoo Finance（股票，15-20分钟延迟）
./run_trading.sh paper --connector yahoo --symbols AAPL,MSFT --interval 10

# 选项2：Binance（加密货币，实时）
./run_trading.sh paper --connector binance --symbols BTCUSDT,ETHUSDT --interval 10

# 选项3：Coinbase Pro（加密货币，实时）
./run_trading.sh paper --connector coinbase --symbols BTC-USD,ETH-USD --interval 10
```

---

## 需要API密钥的连接器

### Alpha Vantage
```bash
export ALPHAVANTAGE_API_KEY='your_key'
./run_trading.sh paper --connector alphavantage --symbols AAPL --interval 60
```

### IEX Cloud
```bash
export IEXCLOUD_API_KEY='your_key'
./run_trading.sh paper --connector iexcloud --symbols AAPL --interval 10
```

### Alpaca
```bash
export ALPACA_API_KEY='your_key'
export ALPACA_API_SECRET='your_secret'
./run_trading.sh paper --connector alpaca --symbols AAPL --interval 5
```

### Polygon.io
```bash
export POLYGON_API_KEY='your_key'
./run_trading.sh paper --connector polygon --symbols AAPL --interval 5
```

---

## 常见问题

**Q: 哪个连接器最适合测试？**
A: Yahoo Finance（无需API密钥）或 Binance Testnet（实时数据）

**Q: 哪个连接器最适合生产环境？**
A: Alpaca（Paper Trading）或 Polygon.io（专业数据）

**Q: 如何切换连接器？**
A: 使用 `--connector` 参数，例如：`--connector yahoo` 或 `--connector alphavantage`

**Q: 连接器之间有什么区别？**
A: 主要区别在于：
- 数据延迟（实时 vs 15-20分钟延迟）
- API调用限制（免费层级）
- 支持的资产类型（股票 vs 加密货币）
- 是否需要API密钥

---

## 技术实现

所有连接器都继承自 `BaseConnector`，实现统一的接口：

```python
class BaseConnector:
    async def connect()
    async def disconnect()
    async def subscribe(symbols: List[str])
    async def unsubscribe(symbols: List[str])
    
    @on_tick
    async def handle_tick(tick: MarketTick)
```

这使得切换连接器非常简单，只需更改 `--connector` 参数即可。








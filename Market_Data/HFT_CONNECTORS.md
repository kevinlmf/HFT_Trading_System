# HFT级数据连接器指南

## ⚠️ 重要：HFT vs 测试连接器

**高频交易（HFT）对延迟要求极高**，需要毫秒级甚至微秒级的数据更新。不是所有连接器都适合HFT。

## 🚀 真正的HFT级连接器

### 1. Alpaca Markets ⭐⭐⭐⭐⭐ **强烈推荐**

**为什么适合HFT：**
- ✅ 实时数据流（毫秒级延迟）
- ✅ WebSocket推送，无需轮询
- ✅ 专业级数据质量
- ✅ 支持订单簿深度数据
- ✅ Paper Trading免费测试

**使用方法：**
```bash
export ALPACA_API_KEY='your_key'
export ALPACA_API_SECRET='your_secret'
./run_trading.sh paper --connector alpaca --symbols AAPL,MSFT --interval 1
```

**适用场景：**
- 股票HFT策略
- 实时订单簿分析
- 低延迟执行

---

### 2. Polygon.io ⭐⭐⭐⭐⭐ **专业级HFT**

**为什么适合HFT：**
- ✅ 机构级数据质量
- ✅ WebSocket实时推送
- ✅ 毫秒级延迟
- ✅ 支持多资产类别（股票、期权、外汇、加密货币）
- ✅ 订单簿深度数据

**使用方法：**
```bash
export POLYGON_API_KEY='your_key'
./run_trading.sh paper --connector polygon --symbols AAPL,MSFT --interval 1
```

**适用场景：**
- 专业HFT交易
- 多资产类别策略
- 订单簿分析

---

### 3. Binance ⭐⭐⭐⭐ **加密货币HFT**

**为什么适合HFT：**
- ✅ WebSocket实时数据流
- ✅ 毫秒级延迟
- ✅ 深度订单簿数据
- ✅ 高流动性市场
- ✅ Testnet免费测试

**使用方法：**
```bash
./run_trading.sh paper --connector binance --symbols BTCUSDT,ETHUSDT --interval 1
```

**适用场景：**
- 加密货币HFT
- 套利策略
- 高频做市

---

### 4. Coinbase Pro ⭐⭐⭐⭐ **加密货币HFT**

**为什么适合HFT：**
- ✅ WebSocket实时数据流
- ✅ 毫秒级延迟
- ✅ 专业级加密货币交易所
- ✅ 订单簿深度数据
- ✅ 高流动性市场
- ✅ Sandbox免费测试，生产环境专业级

**使用方法：**
```bash
# Sandbox测试环境
./run_trading.sh paper --connector coinbase --symbols BTC-USD,ETH-USD --interval 1

# 生产环境（需要API密钥）
export COINBASE_API_KEY='your_key'
export COINBASE_API_SECRET='your_secret'
./run_trading.sh live --connector coinbase --symbols BTC-USD,ETH-USD --interval 1
```

**适用场景：**
- 加密货币HFT
- 专业级交易
- 高频做市
- 套利策略

**注意**: Coinbase Pro是专业的加密货币交易所，生产环境提供机构级低延迟数据。Sandbox环境用于测试，但生产环境完全适合HFT。

---

## ❌ 不适合HFT的连接器

### Yahoo Finance
- **延迟**: 15-20分钟
- **原因**: 延迟太高，完全不适合HFT
- **用途**: 仅用于策略开发和回测

### Alpha Vantage
- **延迟**: REST API，1分钟更新
- **原因**: REST API轮询延迟高，不适合实时交易
- **用途**: 仅用于历史数据分析和测试

### IEX Cloud (免费层级)
- **延迟**: 实时，但有限制
- **原因**: 免费层级消息数限制，可能不够
- **用途**: 测试，付费版适合HFT

---

## HFT连接器对比表

| 连接器 | 延迟 | 数据质量 | HFT适用性 | 推荐度 |
|--------|------|----------|-----------|--------|
| **Alpaca** | ⚡ 毫秒级 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅✅✅✅ |
| **Polygon.io** | ⚡ 毫秒级 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅✅✅✅ |
| **Binance** | ⚡ 毫秒级 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅✅✅✅ |
| **Coinbase Pro** | ⚡ 毫秒级 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅✅✅✅ |
| Yahoo Finance | ⚠️ 15-20分钟 | ⭐⭐ | ❌ | ❌ |
| Alpha Vantage | ⚠️ 1分钟 | ⭐⭐⭐ | ❌ | ❌ |
| IEX Cloud (免费) | ⚡ 实时 | ⭐⭐⭐⭐ | ⚠️ | ⚠️ |

---

## 推荐配置

### 生产环境HFT（股票）
```bash
# 使用Alpaca或Polygon.io
export ALPACA_API_KEY='your_key'
export ALPACA_API_SECRET='your_secret'
./run_trading.sh live --connector alpaca --symbols AAPL,MSFT --interval 1
```

### 生产环境HFT（加密货币）
```bash
# 使用Binance
./run_trading.sh live --connector binance --symbols BTCUSDT,ETHUSDT --interval 1

# 或使用Coinbase Pro（专业级）
export COINBASE_API_KEY='your_key'
export COINBASE_API_SECRET='your_secret'
./run_trading.sh live --connector coinbase --symbols BTC-USD,ETH-USD --interval 1
```

### 测试/开发环境
```bash
# 使用Yahoo Finance（免费，但延迟高）
./run_trading.sh paper --connector yahoo --symbols AAPL --interval 10
```

---

## 延迟要求

**HFT系统延迟要求：**
- **数据接收**: < 10ms
- **信号生成**: < 1ms
- **订单执行**: < 5ms
- **总延迟**: < 20ms

**不适合HFT的连接器延迟：**
- Yahoo Finance: 15-20分钟（900,000-1,200,000ms）❌
- Alpha Vantage: 1分钟（60,000ms）❌

---

## 总结

**真正的HFT应该使用：**
1. ✅ **Alpaca Markets** - 股票HFT首选
2. ✅ **Polygon.io** - 专业级HFT
3. ✅ **Binance** - 加密货币HFT
4. ✅ **Coinbase Pro** - 专业加密货币HFT（生产环境）

**仅用于测试：**
- ❌ Yahoo Finance（延迟太高）
- ❌ Alpha Vantage（REST API延迟）
- ⚠️ IEX Cloud免费版（限制太多）


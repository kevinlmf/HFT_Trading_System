# Yahoo Finance集成说明

## ✅ 已完成的集成

Yahoo Finance连接器已经完全集成到`run_trading.sh`中，可以直接使用。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install yfinance
```

### 2. 运行交易系统

```bash
# 使用Yahoo Finance（免费，无需API密钥）
bash run_trading.sh paper --connector yahoo --symbols AAPL,MSFT --interval 10

# 使用多个标的
bash run_trading.sh paper --connector yahoo --symbols AAPL,MSFT,GOOGL,TSLA --interval 10

# 自定义更新间隔（秒）
bash run_trading.sh paper --connector yahoo --symbols AAPL --interval 5
```

## 📋 功能特性

✅ **完全免费** - 无需API密钥  
✅ **自动数据更新** - 按设定间隔自动获取最新数据  
✅ **QDB集成** - 自动保存数据到QDB  
✅ **策略支持** - 支持所有策略（momentum, mean_reversion等）  
✅ **实时P&L跟踪** - 完整的交易和盈亏跟踪  

## ⚠️ 注意事项

1. **数据延迟**: Yahoo Finance数据有15-20分钟延迟，**不适合实时交易**
2. **适合场景**: 
   - ✅ 策略开发和测试
   - ✅ 回测验证
   - ✅ 学习交易系统
   - ❌ 实时交易（请使用Alpaca或Polygon.io）

3. **更新频率**: 建议设置`--interval`至少10秒，避免过于频繁的API调用

## 🔄 与其他连接器对比

| 连接器 | 免费 | API密钥 | 延迟 | 适用场景 |
|--------|------|---------|------|----------|
| Yahoo Finance | ✅ | ❌ | 15-20分钟 | 测试、开发 |
| Alpaca | ✅ | ✅ | 实时 | Paper Trading |
| Binance | ✅ | ❌ | 实时 | 加密货币 |
| Polygon.io | ✅ | ✅ | 实时 | 专业交易 |

## 📝 示例输出

```
✓ Using Yahoo Finance (no API key needed)
✓ Yahoo Finance connector loaded
✓ Yahoo Finance connector initialized (free, 15-20min delay)
✓ QDB realtime collector initialized
✓ Loaded strategy: momentum
✓ Loaded strategy: mean_reversion
✓ QDB realtime collection started
```

## 🛠️ 故障排除

### 问题1: 找不到yfinance模块
```bash
pip install yfinance
```

### 问题2: 数据更新太慢
- Yahoo Finance本身有延迟，这是正常的
- 可以尝试降低`--interval`值（但不建议低于10秒）

### 问题3: 没有收到数据
- 检查标的代码是否正确（如AAPL, MSFT）
- 确认网络连接正常
- 查看日志中的错误信息

## 🎯 下一步

1. **测试策略**: 使用Yahoo Finance测试你的交易策略
2. **切换到实时数据**: 准备好后，切换到Alpaca或Polygon.io进行实时交易
3. **查看数据**: 数据会自动保存到`./Data/datasets/qdb/`












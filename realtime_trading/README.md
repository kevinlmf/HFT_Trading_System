# Real-Time Trading System

## Overview

The **realtime_trading** module provides a complete real-time trading system that:

1. ✅ **Receives real-time market data** via WebSocket/API
2. 🧠 **Automatically selects the best strategy** based on market conditions
3. 📊 **Generates trading signals** and executes trades
4. 💰 **Tracks P&L in real-time** with comprehensive metrics
5. 📈 **Adapts to market regimes** (trending, mean-reverting, volatile)

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Market Data    │  Alpaca, Binance, IB, etc.
│  API/WebSocket  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  API Connector  │  Real-time tick data
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Strategy Router │  Detects regime → Selects best strategy
│  (Adaptive)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Signal Generator│  Classical, ML, HFT strategies
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Trading Engine  │  Execute trades
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  PnL Tracker    │  Real-time P&L, positions, metrics
└─────────────────┘
```

---

## 📂 Module Structure

```
realtime_trading/
├── connectors/              # Market data connectors
│   ├── base_connector.py   # Abstract base class
│   └── alpaca_connector.py # Alpaca Markets connector
│
├── strategy_router/         # Intelligent strategy selection
│   └── adaptive_router.py  # Auto-select best strategy
│
├── pnl_tracker/            # Real-time P&L tracking
│   └── realtime_pnl.py     # Position & P&L management
│
├── trading_engine.py       # Main trading loop
└── config_example.py       # Configuration template
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install websockets asyncio pandas numpy
```

### 2. Configure API Credentials

```bash
cd realtime_trading/
cp config_example.py config.py
# Edit config.py with your API keys
```

### 3. Run the Trading Engine

```python
import asyncio
from realtime_trading.trading_engine import RealTimeTradingEngine
from realtime_trading.connectors.alpaca_connector import AlpacaConnector
from strategy.classical import MomentumStrategy, PairsTradingStrategy
from strategy.hft_strategies import MarketMakingStrategy

async def main():
    # Initialize connector
    connector = AlpacaConnector(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        paper=True  # Paper trading
    )

    # Initialize strategies
    strategies = {
        'momentum': MomentumStrategy(lookback=20),
        'pairs': PairsTradingStrategy(window=60),
        'mm': MarketMakingStrategy(spread=0.01),
    }

    # Create engine
    engine = RealTimeTradingEngine(
        connector=connector,
        strategies=strategies,
        initial_capital=100000.0,
        symbols=['AAPL', 'MSFT', 'GOOGL']
    )

    # Start trading!
    await engine.start()

asyncio.run(main())
```

---

## 🧠 How It Works

### 1. Market Data Reception

```python
# Real-time WebSocket connection
connector = AlpacaConnector(api_key="...", api_secret="...")

@connector.on_tick
async def handle_tick(tick):
    print(f"{tick.symbol}: ${tick.price} @ {tick.timestamp}")

await connector.start()
await connector.subscribe(['AAPL', 'MSFT'])
```

### 2. Adaptive Strategy Selection

The **AdaptiveStrategyRouter** automatically:
- Detects market regime (trending, mean-reverting, volatile, quiet)
- Tracks each strategy's performance
- Selects the best strategy for current conditions

```python
router = AdaptiveStrategyRouter(strategies={...})

# Auto-select best strategy
strategy_name, confidence = router.select_strategy(market_data)

# Get current regime
regime = router.get_regime_info()
# → {'regime': 'trending', 'confidence': '85%'}
```

**Market Regimes:**
- **Trending**: Momentum, LSTM strategies preferred
- **Mean Reverting**: Pairs trading, stat arb preferred
- **Volatile**: Market making, volatility arbitrage
- **Quiet**: Market making, order flow strategies

### 3. Signal Generation & Execution

```python
# Generate signal from selected strategy
signal = strategy.generate_signal(market_data)

if signal == 1:  # Buy
    engine.execute_buy(symbol, quantity, price)
elif signal == -1:  # Sell
    engine.execute_sell(symbol, quantity, price)
```

### 4. Real-Time P&L Tracking

```python
pnl_tracker = RealTimePnLTracker(initial_capital=100000)

# Open position
pnl_tracker.open_position('AAPL', 100, 150.0, strategy='momentum')

# Update mark-to-market
pnl_tracker.update_market_price('AAPL', 152.0)

# Get metrics
metrics = pnl_tracker.get_performance_metrics()
print(f"Total P&L: {metrics['Total P&L']}")
print(f"Sharpe Ratio: {metrics['Sharpe Ratio']}")
```

---

## 📊 Performance Metrics

The system tracks:

### Trading Metrics
- **Total P&L**: Realized + Unrealized
- **Total Return**: % return on capital
- **Win Rate**: % of winning trades
- **Total Trades**: Number of executed trades

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Current Exposure**: % of capital deployed
- **Commission**: Total transaction costs

### Per-Strategy Metrics
- Individual strategy performance
- Win rate by strategy
- Sharpe ratio by strategy
- Recent returns tracking

---

## 🎯 Market Regime Detection

Uses advanced statistical methods:

1. **Hurst Exponent**
   - H > 0.5: Trending market
   - H < 0.5: Mean-reverting market

2. **Volatility Analysis**
   - Detects high/low volatility regimes

3. **Trend Strength**
   - Linear regression R² for trend quality

4. **Volume Analysis**
   - Confirms regime with volume patterns

---

## ⚙️ Configuration Options

See `config_example.py` for full configuration:

```python
# Trading parameters
INITIAL_CAPITAL = 100000.0
TRADING_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']
MAX_POSITION_SIZE = 0.2  # 20% per position

# Risk management
MAX_DAILY_LOSS = 5000.0
MAX_DRAWDOWN = 0.20
STOP_LOSS_PCT = 0.05

# Strategy selection
ENABLED_STRATEGIES = {
    'momentum': {...},
    'pairs_trading': {...},
    'market_making': {...},
}
```

---

## 🔌 Supported Data Sources

### Currently Implemented
- **Alpaca Markets** (Free tier available)
  - Real-time US stocks via IEX
  - Paper trading supported

### Easy to Add
- **Binance**: Crypto trading
- **Interactive Brokers**: Multi-asset
- **Polygon.io**: Market data
- **Your own WebSocket**: Custom implementation

---

## 📈 Example Output

```
==================================================================
STARTING REAL-TIME TRADING ENGINE
==================================================================
Initial Capital: $100,000.00
Symbols: AAPL, MSFT, GOOGL
Strategies: momentum, pairs_trading, market_making
==================================================================

[2024-01-15 09:30:15] AAPL: $150.25 (Bid: $150.24, Ask: $150.26)
[2024-01-15 09:30:16] Selected strategy: momentum (score: 0.782, regime: trending)
[2024-01-15 09:30:17] Executed BUY AAPL: 66.50 @ $150.25 (Strategy: momentum)

[2024-01-15 09:31:00] P&L Snapshot
Total P&L: $245.50
Equity: $100,245.50
Open Positions: 2

==================================================================
STATUS REPORT
==================================================================
Total P&L: $1,245.50
Total Return: 1.25%
Current Equity: $101,245.50
Total Trades: 23
Win Rate: 60.9%
Sharpe Ratio: 2.15
Max Drawdown: -2.50%
Exposure: 45.2%

Strategy Performance:
  Strategy         Total Trades  Win Rate  Total PnL  Sharpe  Score
  momentum                   12     66.7%    $875.25    2.45  0.842
  pairs_trading               8     50.0%    $245.00    1.82  0.715
  market_making               3     66.7%    $125.25    1.95  0.738

Market Regime: trending
Confidence: 85%
==================================================================
```

---

## 🛡️ Safety Features

1. **Position Size Limits**: Max % of capital per position
2. **Exposure Limits**: Max total market exposure
3. **Stop Loss**: Automatic stop-loss orders
4. **Daily Loss Limit**: Stop trading if max loss exceeded
5. **Drawdown Protection**: Halt trading on excessive drawdown
6. **Trading Hours**: Respect market hours
7. **Graceful Shutdown**: Closes all positions on exit

---

## 🧪 Testing

```bash
# Test with paper trading
python examples/test_realtime_trading.py --paper

# Backtest mode (historical data)
python examples/test_realtime_trading.py --backtest --start 2024-01-01 --end 2024-12-31
```

---

## 📚 Further Reading

- **[Main README](../README.md)** - System overview
- **[Strategy Module](../strategy/README.md)** - Available strategies
- **[Execution Module](../execution/README.md)** - Performance optimization

---

## ⚠️ Disclaimer

**FOR PAPER TRADING AND TESTING ONLY**

This real-time trading system is for educational purposes. Always:
- Start with paper trading
- Test thoroughly before live trading
- Understand the risks
- Never trade with capital you can't afford to lose

**Use at your own risk. No guarantees of profitability.**

---

## 🤝 Contributing

To add new data connectors:

1. Inherit from `BaseConnector`
2. Implement `connect()`, `subscribe()`, `disconnect()`
3. Emit ticks via `self._emit_tick(tick)`

```python
from connectors.base_connector import BaseConnector, MarketTick

class MyConnector(BaseConnector):
    async def connect(self):
        # Your connection logic
        pass

    async def subscribe(self, symbols):
        # Your subscription logic
        pass
```

---

*Trade smart, trade fast, trade algorithmically!* 🚀

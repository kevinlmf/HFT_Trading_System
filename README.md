## HFT_Trading_System

**HFT_Trading_System** is designed to answer two fundamental questions:

1. **How can we make trading faster?**
   → Hardware-efficient C++ execution (<1µs latency), lock-free data pipelines, and optimized memory design.

2. **What are the best strategies to trade?**
   → Data-driven discovery using reinforcement learning, deep learning, and statistical inference.

### Setup Environment
```bash
git clone https://github.com/kevinlmf/HFT_Trading_System
cd HFT_Trading_System
```

## Quick Start

```bash
# 1. Build entire system (CPU + optional CUDA)
./build_system.sh --all --test

# 2. Set API keys (for paper/live trading)
export ALPACA_API_KEY='your_key'
export ALPACA_API_SECRET='your_secret'

# 3. Run trading system
./run_trading.sh demo                    # Demo mode (market data only)
./run_trading.sh paper --dashboard       # Paper trading (no real money)
./run_trading.sh backtest --dashboard    # Backtest with monitoring

# Advanced usage
./run_trading.sh paper --symbols AAPL,TSLA --strategies momentum,market_making
./run_trading.sh backtest --capital 50000 --strategies all
./run_trading.sh monitor-only --dashboard  # Monitoring only
```

**Installation**: Python 3.8+, GCC 7.0+ (C++17), CUDA 11+ (optional)
```bash
pip install numpy pandas torch arch statsmodels scipy matplotlib flask websockets plotly dash
```

## From Research to Production: End-to-End Trading System Lifecycle

## System Abstraction: Data → Environment → Execution → Monitoring

| **Stage** | **Core Function** | **Actual Components** | **Output** |
|:-----------|:------------------|:----------------------|:------------|
| **1. Data Layer** | Real-time market data collection | Alpaca WebSocket Connector, Market Tick & OrderBook Data Structures | Real-time Prices, Bid/Ask Spreads, Volume |
| **2. Environment Layer** | Simulate and validate trading strategies | Realistic Market Simulator (slippage/latency/partial fills), Backtesting Engine, Market Regime Detector (5 states) | Simulated Trades, Historical Performance, Market Regimes |
| **3. Execution Layer** | Multi-strategy trading with adaptive routing and risk control | 11 Strategies (Classical/ML/RL/HFT), Adaptive Strategy Router, Portfolio Risk Manager (6 models, 7 constraints), C++ Data Feed | Approved Orders, Position Updates, Risk Metrics |
| **4. Monitoring Layer** | Performance tracking and analysis | Real-time PnL Tracker, Performance Metrics Calculator (30+ metrics), Throughput Benchmarks | PnL Reports, Risk/Return Metrics, System Logs |


### Complete System Flow

```
Market Data (Real-time via Alpaca WebSocket)
    |
    v
[Data Connectors] → Market Ticks & Order Book Snapshots
    |
    v
[Market Regime Detector] → Classify: Trending / Mean-Reverting / Volatile / Quiet / Unknown
    |
    v
[Adaptive Strategy Router] → Score & Select Best Strategy for Current Regime
    |
    v
[Strategy Execution]
    ├─ Classical (4): Momentum, Mean-Variance, Pairs Trading, Statistical Arbitrage
    ├─ ML-Based (6): LSTM, Transformer, CNN, Random Forest, XGBoost, LightGBM
    ├─ RL-Based (6): DQN, A3C, PPO, A2C, SAC, TD3
    └─ HFT (2): Market Making (inventory skewing), Order Flow Imbalance
    |
    v
[Generate Orders] → Buy/Sell/Hold signals with position sizes
    |
    v
[Portfolio Risk Manager]
    ├─ Risk Models: Equal-weight, Inverse Vol, Mean-Variance, Risk Parity, Black-Litterman, HRP
    └─ 7 Constraints: Position Size (20%), Sector (40%), Correlation (60%),
                      Volatility (15%), Drawdown (5%), VaR (2%), Concentration (50% in top 5)
    |             |
    v             v
APPROVED      REJECTED
    |             |
    v             v
[Market Simulator]  Block & Log
(Spreads, Slippage,    |
 Latency, Partial      v
 Fills, Impact)     Alert User
    |
    v
Execute Trade → Update Position
    |
    v
[Real-time PnL Tracker]
    ├─ Realized P&L (closed positions)
    ├─ Unrealized P&L (open positions, mark-to-market)
    ├─ Equity Curve (with high water mark)
    └─ Commission Tracking
    |
    v
[Performance Metrics Calculator]
    ├─ Returns: Total, Annualized, Cumulative
    ├─ Risk: Volatility, Max Drawdown, VaR (95%), CVaR (95%)
    ├─ Risk-Adjusted: Sharpe, Sortino, Calmar, Omega
    ├─ Trading Stats: Win Rate, Profit Factor, Avg Win/Loss, Trade Duration
    └─ Distribution: Skewness, Kurtosis, Best/Worst Day/Month
    |
    v
[Backtesting & Analysis] → Feedback to Strategy Router (Regime-Performance Mapping)
```
## Directory Structure
```
HFT_System/
├── Data/                  # Real-time data layer
│   ├── connectors/        # Alpaca WebSocket connector
│   ├── datasets/          #  ⚠️Historical loaders
│   └── preprocessors/     #  ⚠️ Feature engineering
│
├── Environment/           # Simulation & validation
│   ├── simulator/         # Market simulator (spread, slippage, latency)
│   ├── backtester/        # Historical backtesting with metrics
│   └── pricing/           # ⚠️ Option pricing models
│
├── Execution/             # Strategy execution & risk control
│   ├── strategies/        # 18 total strategies
│   │   ├── classical/     # Momentum, Mean-Variance, StatArb
│   │   ├── ml_based/      # LSTM, Transformer, XGBoost (+ RL)
│   │   └── hft_strategies/# Market Making, Order Flow Imbalance
│   ├── risk_control/      # Portfolio manager, CVaR models
│   ├── trading/           # Engine + Adaptive Strategy Router
│   └── cpp_core/          # C++ data feed (PyBind11 bindings)
│
├── Monitoring/            # Performance tracking
│   ├── evaluation/        # 30+ metrics (returns, risk, distribution)
│   ├── benchmarks/        # Throughput & latency tests
│   ├── pnl_tracking/      # Real-time P&L tracker
│   └── dashboard/         # ⚠️ WebSocket dashboard
│
├── cuda_accelerated/      # GPU acceleration
│   ├── kernels/           # CUDA kernels (backtest, risk calc)
│   └── python/            # Python bindings for CUDA modules
│
├── examples/              # Example workflows
├── scripts/               # Evaluation & comparison scripts
├── build_system.sh        # Build script (C++ + CUDA)
└── run_trading.sh         # System launcher (demo/backtest/live)

**Legend**: ⚠️ Directory exists but empty (reserved for future development)
```
---



## Configuration

### Adjusting Strategy Parameters
```python
# HFT Market Making Strategy
from Execution.strategies.hft_strategies import market_making
strategy = market_making.MarketMakingStrategy(
    spread=0.01,              # Bid-ask spread (1 bp)
    inventory_limit=1000,     # Max position size
    order_size=100,           # Default order size
    risk_aversion=0.5         # Risk aversion parameter
)

# Machine Learning Strategy
from Execution.strategies.ml_based import rl_strategies
rl_strategy = rl_strategies.PPOStrategy(
    learning_rate=3e-4,       # Learning rate
    gamma=0.99,               # Discount factor
    clip_epsilon=0.2,         # PPO clip parameter
    update_epochs=10          # Training epochs per update
)
```

---
## Performance Metrics

### Latency (C++ Core) - **Answering: How fast can we trade?**
| Operation | Latency | Technique |
|:----------|:--------|:----------|
| Order Processing | <1 µs | Lock-free queues, zero-copy memory |
| Risk Checks | <5 µs | Pre-computed Greeks, SIMD operations |
| Order Book Updates | <10 µs | Custom allocators, cache-aligned data |

### Throughput (CUDA) - **Enabling overnight strategy discovery**
| Task | Performance | Speedup |
|:-----|:------------|:--------|
| Backtest (1000 strategies) | 30 sec vs 25 min CPU | 50x |
| Monte Carlo VaR (1M paths) | 2 ms vs 1.2 sec CPU | 250x |
| Greeks Calculation (1000 options) | 3 ms vs 500 ms CPU | 167x |

---

## Future Research Directions

1. **Statistical Computing Optimization**
Designing algorithms with **lower space and time complexity**: numerical linear algebra optimizations, memory-efficient data structures, distributed computing frameworks for ultra-high-frequency settings.

2. **Microstructure-Aware Strategy Adaptation**
Developing **adaptive HFT strategies** responsive to evolving **market microstructures**: order flow imbalance, spread dynamics, latency arbitrage.

---

## Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**
This software is **not intended for live trading** without extensive testing.
Trading involves substantial risk. Past performance does not guarantee future results.

---

## Acknowledgments

Built using: [Pybind11](https://github.com/pybind/pybind11), [PyTorch](https://pytorch.org/), [JAX](https://github.com/google/jax), [NumPy](https://numpy.org/), [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

**License**: MIT

---

When curiosity meets motion, every millisecond holds the spark.🌅

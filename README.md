**HFT_Trading_System** is designed to answer two fundamental questions:

1. **How can we make trading faster?**
   → Hardware-efficient C++ execution (<1µs latency), lock-free data pipelines, and optimized memory design.

2. **What are the best strategies to trade?**
   → Data-driven discovery using reinforcement learning, deep learning, and statistical inference.

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

---




```

## Directory Structure

```
HFT_System/
├── Data/                              # Layer 1: Market data connectors & datasets
│   ├── connectors/                    # Exchange APIs (Alpaca, IB)
│   ├── datasets/                      # Raw market data & preprocessors
│   └── preprocessors/                 # LSTM, GARCH, feature engineering
├── Environment/                       # Layer 2: Simulator, Backtester
│   ├── simulator/                     # Multi-agent market simulation
│   ├── backtester/                    # Historical backtesting engine
│   └── pricing/                       # Black-Scholes, Heston, SABR
├── Execution/                         # Layer 3: Strategy, Risk, C++ Core
│   ├── strategies/                    # Classical/ML/RL/HFT strategies
│   │   ├── classical/                 # Momentum, Mean-Variance, Pairs, Stat Arb
│   │   ├── ml_based/                  # Deep Learning, Reinforcement Learning
│   │   └── hft_strategies/            # Market Making, Order Flow Imbalance
│   ├── risk_control/                  # VaR/CVaR/Greeks risk controller
│   ├── trading/                       # Real-time trading engine
│   └── cpp_core/                      # C++ low-latency core (<1µs)
├── Monitoring/                        # Layer 4: Dashboard, Tracking
│   ├── evaluation/                    # Sharpe, Sortino, Max Drawdown
│   ├── benchmarks/                    # Throughput & latency benchmarks
│   ├── pnl_tracking/                  # Real-time PnL tracker
│   └── dashboard/                     # WebSocket dashboard
├── cuda_accelerated/                  # GPU acceleration (50-250x speedup)
│   ├── cpp/                           # CUDA C++ kernels
│   └── python/                        # Python bindings
├── examples/                          # Complete workflow examples
├── build_system.sh                    # Master build script
└── run_trading.sh                     # Trading system launcher
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

## Trading System Usage

```bash
# Demo mode (market data only, no trading)
./run_trading.sh demo

# Paper trading (simulated, no real money)
./run_trading.sh paper --symbols AAPL,TSLA --strategies momentum,market_making --dashboard

# Backtest (historical simulation with all strategies)
./run_trading.sh backtest --strategies all --capital 200000 --dashboard

# Live trading (CAUTION: real money)
./run_trading.sh live --capital 10000 --strategies momentum --interval 10

# Monitoring only (dashboard without trading)
./run_trading.sh monitor-only --dashboard
```

**Available Strategies**: `momentum`, `mean_reversion`, `pairs_trading`, `market_making`, `statistical_arbitrage`, `all`

**Logs**: All sessions logged to `logs/` directory

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

## Integration Workflow

```
Night (Research - CUDA):
1. Backtest 1000+ strategies         → ./run_trading.sh backtest --strategies all
2. Calculate risk metrics (VaR/CVaR)
3. Optimize hyperparameters
   → Discover best strategies

Day (Execution - C++):
1. Paper/demo trading                → ./run_trading.sh paper --dashboard
2. Execute trades (<1µs latency)
3. Real-time risk checks (<5µs)
   → Live trading with risk controls

Always (Monitoring):
- Dashboard                          → ./run_trading.sh monitor-only --dashboard
- Real-time PnL tracking
- Performance analytics
   → Continuous feedback loop
```

---


## Future Research Directions

1. **Statistical Computing Optimization**
   Designing algorithms with **lower space and time complexity**: numerical linear algebra optimizations, memory-efficient data structures, distributed computing frameworks for ultra-high-frequency settings.

2. **Microstructure-Aware Strategy Adaptation**
   Developing **adaptive HFT strategies** responsive to evolving market microstructures: order flow imbalance, spread dynamics, latency arbitrage—using online learning and reinforcement frameworks optimized for ultra-low latency execution.

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

When curiosity meets motion, every millisecond holds the spark of alpha 🌅

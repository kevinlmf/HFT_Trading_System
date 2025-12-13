# High-Frequency Trading System (HFT)

A full-stack HFT research and execution platform integrating latency engineering, strategy intelligence, and comprehensive HFT metrics evaluation.

## System Architecture

The system follows a streamlined HFT pipeline optimized for low-latency execution:

```
Market Data
    ↓
LOB Processing (Order Book Analysis)
    ↓
Microstructure Signals
    ↓
Alpha / Prediction
    ↓
Strategy Logic
    ↓
Execution Engine
    ↓
Fill Feedback Loop
    ↓
Risk Limits
    ↓
Monitoring
```

### Pipeline Components

1. **Market Data** - Real-time market data ingestion from multiple sources
2. **LOB Processing** - Order book analysis and depth processing
3. **Microstructure Signals** - Market microstructure feature extraction
4. **Alpha / Prediction** - Signal generation and prediction models
5. **Strategy Logic** - Trading strategy implementation
6. **Execution Engine** - Smart order execution with slippage minimization
7. **Fill Feedback Loop** - Execution feedback for strategy refinement
8. **Risk Limits** - Real-time risk monitoring and position limits
9. **Monitoring** - Performance tracking and HFT metrics evaluation

## Key Features

### Trading Strategies and Factor discover

The system includes a comprehensive suite of trading strategies for evaluation and comparison:

**Traditional Strategies:**
- **Momentum** - Trend-following strategy based on price momentum over lookback periods
- **Mean Reversion** - Counter-trend strategy that trades when prices deviate from moving averages

**Machine Learning Strategies:**
- **ML Random Forest** - Random Forest classifier trained on technical features (rolling returns, volatility, price-to-MA ratios)
- **ML XGBoost** - Gradient boosting classifier with enhanced feature engineering and probability-based signal generation

**Reinforcement Learning Strategies:**
- **RL Simple** - Simplified RL policy based on state-value functions and feature thresholds

**LLM-Enhanced Strategies:**
- **LLM Sentiment** - Pattern-based sentiment analysis using price trend detection
- **LLM Pattern** - Technical pattern recognition (head-and-shoulders, trend reversals)

All strategies are automatically evaluated in `complete-flow` mode with comprehensive HFT metrics (Hit Ratio, Latency Jitter, Slippage, Throughput, etc.), and the best-performing strategy is automatically selected based on Sharpe ratio.

## How Speed Is Achieved

The system builds a **multi-layered latency core**, optimized from rapid prototyping to large-scale execution:

- **QDB (Quantitative Database)** – **5–20x overall speedup** for data access via:
  - **O(log n) indexing** with binary search (vs O(n) linear scan)
  - **Parallel file loading** for multi-symbol queries
  - **LRU cache** with large gains on repeated queries
  - **Memory mapping** for multi-process data sharing

- **Vectorized Linear Algebra (NumPy / pandas)** – Most research, backtesting, risk and HFT metrics are implemented as **batched array ops and linear algebra** (returns, covariances, factor transforms), enabling **sub‑millisecond prototyping** when data fits in memory cache.

- **Pybind11 C++ Core** – High‑throughput C++ slippage / execution engine (`Execution/cpp_core`), benchmarked vs Python/NumPy (see `benchmark-slippage`) to ensure the core can reach HFT‑level throughput.

- **Smart Executor** – Dynamically routes workloads between pure Python, NumPy vectorized paths and the C++ core based on data size and availability, so small experiments stay lightweight while large runs automatically use the fast path.

- **Optimization Stack (NumPy + JAX backends)** – Parallelizes risk metric and objective evaluation (Sharpe, CVaR, volatility, portfolio optimization). Optional **JAX‑based optimizers / deep models** (in `Optimization/jax_optimizer.py` and `Strategy_Construction/ml_based/`) can JIT‑compile heavy numerical kernels when a GPU/TPU is available, further reducing end‑to‑end pipeline runtime.


### HFT Metrics Evaluation

The system includes comprehensive HFT-specific metrics:

- **Hit Ratio** - Signal directional accuracy (target: >55%)
- **Latency Jitter** - Processing delay variance (target: <2ms)
- **Cancel-to-Trade Ratio** - Order cancellation metrics
- **Order Book Imbalance Importance** - LOB feature significance
- **Alpha Decay** - Signal validity period (typically 5-50ms for HFT)
- **Slippage** - Execution cost in basis points (target: <1 bps)
- **Throughput (TPS)** - Orders processed per second (target: 1000-10000 TPS)

All metrics are automatically calculated and saved to `results/hft_metrics/` with detailed reports.

In addition, the **Research Framework + complete-flow** pipeline automatically exports:
- `results/research/research_complete_flow_*.json` – full microstructure profiling, factor hypotheses, and validation results.
- `results/research/factor_attribution_*.json` – a **factor-level alpha attribution summary**, ranking the most important microstructure factors (by IC / t‑stat / Sharpe when available) that contribute to the observed alpha.

## HFT Metrics Benchmark

### HFT Metrics vs Benchmarks

- **HFT metrics** answer: *“How good is this HFT strategy in live / backtest trading?”*  
  - Computed from real trading or `complete-flow` runs  
  - Saved to `results/hft_metrics/`  
  - Include Hit Ratio, Latency Jitter, Cancel-to-Trade Ratio, Alpha Decay, Slippage, Throughput, etc.

- **Benchmarks** answer: *“How fast are the core components (Python vs NumPy vs C++)?”*  
  - Run via `./run_trading.sh benchmark-slippage`  
  - Saved to `results/benchmarks/` as JSON + PNG  
  - Compare raw execution time, throughput, and speed-up of the slippage engine across implementations.

Together, they connect **strategy-level HFT performance** (HFT metrics) with **infrastructure-level speed** (benchmarks):  
the benchmarks act as **evidence and capacity tests** that the underlying engine is fast enough to support the target HFT metrics (Hit Ratio, Latency, Slippage, TPS) in real trading.

### Target Performance Levels (HFT Metrics)

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **Hit Ratio** | >55% | 50-55% | 45-50% | <45% |
| **Latency Jitter** | <0.5ms | 0.5-2ms | 2-10ms | >10ms |
| **Alpha Decay** | 5-20ms | 20-50ms | 50-200ms | >200ms |
| **Slippage** | <1 bps | 1-3 bps | 3-5 bps | >5 bps |
| **Throughput** | >10K TPS | 1K-10K TPS | 100-1K TPS | <100 TPS |
| **Cancel-to-Trade** | 10-50 | 0.1-1.0 | 1-10 | >100* |

*High cancel-to-trade ratios may attract regulatory attention

See [Evaluation/hft_benchmarks.md](Evaluation/hft_benchmarks.md) for detailed benchmarks.


## Project Structure

```
HFT_System/
├── Market_Data/                    # Market data connectors
│   ├── alpaca_connector.py
│   ├── polygon_connector.py
│   ├── binance_connector.py
│   └── ...
├── QDB/                            # Quantitative Database
│   ├── qdb.py
│   ├── indexer.py
│   ├── ingestion.py
│   └── ...
├── Microstructure_Analysis/       # LOB processing & microstructure signals
│   └── microstructure_profiling.py
├── Alpha_Modeling/                 # Alpha generation & prediction
│   ├── factor_hypothesis.py
│   ├── statistical_validation.py
│   └── ml_validation.py
├── Strategy_Construction/         # Strategy logic
│   ├── base_strategy.py
│   ├── classical/
│   ├── ml_based/
│   └── hft_strategies/
├── Execution/                      # Execution engine
│   ├── engine/
│   │   ├── smart_executor.py
│   │   └── integrated_trading_flow.py
│   └── trading/
│       └── trading_engine.py
├── Evaluation/                     # Performance evaluation
│   ├── hft_metrics.py             # HFT-specific metrics
│   ├── performance_metrics.py
│   └── strategy_benchmark.py
├── Optimization/                   # Strategy & execution optimization
│   ├── hft_optimizer.py           # HFT optimization tools
│   └── optimization_stack.py
├── Risk_Control/                   # Risk limits & monitoring
│   ├── risk_metrics.py
│   └── portfolio_manager.py
├── Monitoring/                     # Real-time monitoring & benchmarks
│   ├── realtime_pnl.py
│   └── benchmarks/                # Slippage & throughput benchmarks
├── results/                        # Auto-generated outputs
│   ├── hft_metrics/               # HFT metrics reports
│   ├── performance/               # Complete-flow results
│   ├── strategies/                # Strategy comparison summaries
│   ├── backtest/                  # Backtest summaries
│   ├── research/                  # Research framework outputs
│   └── benchmarks/                # Speed benchmarks (Python vs C++ etc.)
└── run_trading.sh                  # Main execution script
```

## Quick Start
```bash
git clone https://github.com/kevinlmf/HFT_Trading_System
cd HFT_Trading_System
```
### Basic Usage

```bash
# Complete trading flow with HFT metrics evaluation
./run_trading.sh complete-flow --symbols AAPL,MSFT

# Paper trading with real-time data
./run_trading.sh paper --symbols AAPL,MSFT

# Paper trading with Yahoo Finance (free, 15-20min delay)
./run_trading.sh paper --connector yahoo --symbols AAPL,MSFT --interval 10

# Enable HFT optimization
ENABLE_HFT_OPTIMIZATION=true ./run_trading.sh complete-flow --symbols AAPL,MSFT

# Run slippage performance benchmark (Python vs NumPy vs C++)
./run_trading.sh benchmark-slippage
```

### Output

All results are automatically saved to `results/`:

- **`results/hft_metrics/`** - HFT performance metrics and detailed reports
  - `hft_metrics_*.json` - Complete HFT metrics for all strategies (Hit Ratio, Latency Jitter, Cancel-to-Trade Ratio, Alpha Decay, Slippage, Throughput)
  - `report_<strategy>_*.txt` - Human-readable performance reports per strategy

- **`results/performance/`** - Complete-flow execution results
  - `complete_flow_*.json` - End-to-end pipeline diagnostics, best strategy selection, and overall performance summary

- **`results/strategies/`** - Strategy comparison and ranking
  - `strategy_comparison_*.json` - Per-strategy returns, Sharpe ratios, volatility, and risk metrics for all evaluated strategies

- **`results/backtest/`** - Backtest execution summaries
  - `backtest_results_*.json` - Extracted backtest results with PnL, drawdown, and trade statistics

- **`results/research/`** - Research framework outputs (microstructure analysis & factor discovery)
  - `research_complete_flow_*.json` - Full microstructure profiling results, factor hypotheses, and statistical validation
  - `factor_attribution_*.json` - Factor-level alpha attribution summary, ranking the most important microstructure factors by IC, t-stat, and Sharpe ratio

- **`results/benchmarks/`** - Performance benchmarks
  - Speed comparison results (Python vs NumPy vs C++ slippage performance) saved as JSON and visualization PNGs

## Data Connectors

| Connector | Free | API Key | Latency | HFT Ready | Best For |
|-----------|------|---------|---------|-----------|----------|
| **Alpaca** | Yes | Yes | Real-time | 5/5 | **HFT Trading** |
| **Polygon.io** | Yes | Yes | Real-time | 5/5 | **Professional HFT** |
| **Binance** | Yes | No | Real-time | 4/5 | **Crypto HFT** |
| **Coinbase Pro** | Yes | No | Real-time | 4/5 | **Crypto HFT** |
| **Yahoo Finance** | Yes | No | 15-20 min delay | No | Testing Only |

**Recommended for HFT**: Alpaca, Polygon.io, Binance, Coinbase Pro






---
## Safety, Risk & Licensing Notice

This project is provided **for research and educational purposes only**. It is **not intended for live trading or commercial deployment** without
extensive independent validation, stress testing, and risk review. Trading financial instruments involves substantial risk.
**Past performance is not indicative of future results.* The authors assume no responsibility for any financial losses incurredthrough the use of of this code.

---
When curiosity meets motion, every millisecond holds the spark🌄

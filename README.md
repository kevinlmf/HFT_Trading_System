# High-Frequency Trading System (HFT)

A full-stack HFT research and execution platform integrating latency engineering, strategy intelligence, and a five-layer optimization stack.

## System Architecture

```
Market Data Connectors
    ↓
    ├─→ QDB (Quantitative Database) ─→ [Data Storage & Versioning]
    │
    └─→ Market Microstructure Analysis (Research Framework)
            ↓
        Strategy Selection (based on microstructure insights)
            ↓
        Smart Execution → Strategy Evaluation → Optimization Stack → Risk Control → Portfolio Management → Monitoring & Reporting
        
        
```

## Key Features

- **QDB (Quantitative Database)** – Unified data layer with O(log n) indexing, LRU caching (500x speedup), and versioning. **5-20x overall speedup**, <10ms query target.

- **Research Framework** – Market microstructure profiling → economics-motivated factor hypothesis → statistical/ML validation. Correct paradigm: Economics → Statistics → Algorithms.
- **Quick Validation Layer** – Lightweight backtest + Monte Carlo validation (< 100ms) before trade execution. Caching (60s TTL) and timeout protection (500ms) ensure low latency while maintaining safety.
- **Performance Monitoring** – Comprehensive latency tracking: tick processing, signal generation, validation times. Final reports include Net P&L, Sharpe ratio, throughput metrics, and detailed performance statistics.
- **Performance** – QDB: 5-20x speedup. C++ Core: <1 µs per order, 100M+ orders/sec. CUDA: 200M+ orders/sec for Monte Carlo and bulk operations. Validation: < 100ms with caching.



**For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)**

## Project Structure

```
HFT_System/
├── Data/
│   ├── qdb/                            # Quantitative Database (QDB) - unified data layer
│   │   ├── qdb.py                      # Main QDB class (integrates all components)
│   │   ├── optimized_indexer.py       # O(log n) indexing with parallel loading
│   │   ├── cache.py                    # LRU cache with memory mapping (500x speedup)
│   │   └── ingestion.py               # Real-time and historical data collection
│   └── connectors/                     # Market data connectors
│       ├── alpaca_connector.py         # Alpaca Markets (real-time stocks)
│       ├── binance_connector.py        # Binance (cryptocurrency)
│       ├── polygon_connector.py        # Polygon.io (professional data)
│       ├── coinbase_connector.py       # Coinbase Pro (cryptocurrency)
│       └── yahoo_connector.py          # Yahoo Finance (free, 15-20min delay)
├── Research/                           # Quantitative Research Framework
├── Execution/                          # Trading engine, strategies, risk control
├── Optimization/                       # Five-layer optimization stack
├── Monitoring/benchmarks/              # Latency and statistics benchmarks
├── Execution/cpp_core/                 # Pybind11 C++ low-latency core
└── results/                            # Auto-generated strategy & latency outputs
```

## Why Speed Matters

In high-frequency trading, **latency defines profitability**. Every microsecond of delay can mean losing queue priority, missing a fill, or mispricing risk. Speed is not a feature — it is the foundation that makes statistical intelligence executable in real markets.

## How Speed Is Achieved

The system builds a **multi-layered latency core**, optimized from rapid prototyping to large-scale execution:

- **QDB (Quantitative Database)** – **5-20x overall speedup** through:
  - **O(log n) indexing** with binary search (vs O(n) linear scan)
  - **Parallel file loading** for multi-symbol queries
  - **LRU cache** with 500x speedup for repeated queries
  - **Memory mapping** for multi-process data sharing
  - **Unified data layer** ensuring consistency across paper/backtest/live modes

- **Python Vectorized Path** – Enables sub-millisecond prototyping when datasets fit cache, perfect for model iteration and diagnostics.

- **Pybind11 C++ Core** – <1 µs per order, >100M orders/sec throughput for latency-critical slippage and execution computations.

- **CUDA Accelerators** – >200M orders/sec for Monte Carlo simulations, large-scale backtests, and bulk slippage pricing.

- **Smart Executor** – Dynamically routes workloads and supports deterministic benchmarking via `--force-slippage-impl`.

- **Optimization Stack** – Parallelizes risk metric and objective evaluation (Sharpe, CVaR, volatility), reducing total pipeline runtime while enhancing analytic depth.


## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/kevinlmf/HFT_Trading_System
cd HFT_Trading_System

# Make helper scripts executable (first run only)
chmod +x build_system.sh run_trading.sh

# Optional: build C++/CUDA components and run smoke tests
./build_system.sh --all --test

# Launch the complete trading flow (EDA → strategies → risk/positions → reports)
./run_trading.sh complete-flow --symbols AAPL,MSFT,GOOGL

# Paper trading with QDB (default enabled, automatic data collection)
./run_trading.sh paper --symbols AAPL,MSFT

# Paper trading with QDB optimization (O(log n) indexing, parallel loading)
./run_trading.sh paper --symbols AAPL,MSFT --qdb-optimized

# Paper trading with Yahoo Finance (FREE, no API key needed!)
./run_trading.sh paper --connector yahoo --symbols AAPL,MSFT --interval 10


# Additional modes
./run_trading.sh paper --dashboard        # Paper trading with live dashboard
./run_trading.sh backtest --dashboard     # Backtest with visual monitoring
./run_trading.sh benchmark-slippage       # Python vs C++ vs CUDA latency benchmark
```




## Current Limitations

- Synthetic market data still produces extreme returns for certain strategies; further calibration is required for production realism.
- C++ and CUDA builds depend on toolchain availability; users without compilers fall back to Python paths.
- GPU acceleration is optional and disabled when CuPy/CUDA is missing; dynamic detection adds slight startup overhead.
- Risk checks remain sensitive to negative Sharpe ratios; tolerances are relaxed but may still flag high-volatility regimes.

---

# Safety & Licensing

**For research and education only.** Live deployment demands extensive validation. Trading involves risk; past performance is not indicative of future results.

License: MIT

---

When curiosity meets motion, every millisecond holds the spark🌄

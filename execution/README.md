# Execution Module

[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

The **execution** module is the performance-critical core of the HFT system, responsible for **ultra-low-latency order execution, market simulation, performance benchmarking, and strategy evaluation**. This module addresses the fundamental question: **"How can we make trading faster?"**

## Core Components

### 1. C++ Execution Engine (`cpp_core/`)

High-performance C++ implementation with Python bindings for sub-microsecond order processing.

**Key Features:**
- Sub-microsecond order processing with lock-free data structures
- SIMD vectorization and cache-aligned memory layouts
- Zero-copy data transfer with Pybind11 bindings
- JAX integration support

### 2. Market Simulator (`engine/`)

Realistic market simulation engine modeling real-world trading conditions.

**Features:**
- Multiple order types (Market, Limit, Stop, Stop-Limit)
- Realistic slippage, partial fills, market impact
- Latency modeling and spread dynamics
- Volatility regimes and liquidity modeling

### 3. Performance Benchmarks (`benchmarks/`)

Performance testing suite measuring throughput and latency.

**Metrics:**
- Order processing latency (p50, p95, p99)
- System throughput (orders/second)
- Memory usage and CPU utilization

### 4. Strategy Evaluation (`evaluation/`)

Advanced evaluation framework for analyzing strategy performance.

**Metrics:**
- Returns: Total, annualized, CAGR
- Risk-Adjusted: Sharpe, Sortino, Calmar ratios
- Risk: Max drawdown, volatility, VaR, CVaR
- Trade Analytics: Win rate, profit factor
- P&L Attribution: Gross P&L, commissions, slippage

## Module Structure

```
execution/
├── cpp_core/                  # C++ execution engine
│   ├── src/                  # C++ source files
│   │   ├── order.cpp         # Order management
│   │   └── data_feed.cpp     # Market data feed
│   ├── include/              # Header files
│   │   ├── order.hpp         # Order definitions
│   │   ├── fast_orderbook.hpp # Lock-free order book
│   │   ├── data_feed.h       # Data feed interface
│   │   └── order_executor.hpp # Execution logic
│   ├── bindings/             # Python bindings
│   │   └── all_bindings.cpp  # Pybind11 interface
│   ├── jaxbind/              # JAX integration
│   │   └── bindings.cpp      # JAX bindings
│   └── setup.py              # Build configuration
│
├── engine/                   # Execution logic
│   └── market_simulator.py   # Market simulation engine
│
├── benchmarks/               # Performance testing
│   ├── benchmark_throughput.py
│   └── visualize_benchmark.py
│
├── evaluation/               # Performance metrics
│   ├── __init__.py
│   ├── performance_metrics.py
│   ├── strategy_evaluator.py
│   └── pnl_analyzer.py
│
├── build_cpp.sh             # Build script
└── test_build.py            # Build verification
```

## Installation & Setup

### Building the C++ Core

From the execution directory:

```bash
# Using build script (recommended)
./build_cpp.sh

# Manual build
cd cpp_core
python setup.py build_ext --inplace
cd ..

# Verify build
python test_build.py
```

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Python 3.10+
- Pybind11
- NumPy
- JAX (optional, for JAX bindings)

## Usage

### 1. Running Performance Benchmarks

```bash
# Throughput benchmark
python benchmarks/benchmark_throughput.py

# Visualize benchmark results
python benchmarks/visualize_benchmark.py
```

Expected performance:
- **Order Processing**: < 1 microsecond (p50)
- **Throughput**: 1M+ orders/second
- **Latency**: < 5 microseconds (p99)

### 2. Using the Market Simulator

```python
from execution.engine.market_simulator import MarketSimulator, OrderType, OrderSide

simulator = MarketSimulator(initial_capital=100000, commission_rate=0.001)
order = simulator.create_order("AAPL", OrderSide.BUY, OrderType.LIMIT, 100, 150.50)
trades = simulator.execute(order, market_data)
```

### 3. Evaluating Strategy Performance

```python
from execution.evaluation import StrategyEvaluator, PnLAnalyzer

evaluator = StrategyEvaluator(risk_free_rate=0.02)
report = evaluator.evaluate("MyStrategy", returns_series, trades_df)
print(f"Sharpe: {report['sharpe_ratio']:.2f}, Drawdown: {report['max_drawdown']:.2%}")

pnl_analyzer = PnLAnalyzer()
breakdown = pnl_analyzer.analyze_pnl(trades_df)
```

### 4. Using C++ Execution Engine

```python
import sys
sys.path.insert(0, 'execution/cpp_core')
import cpp_trading2

orderbook = cpp_trading2.FastOrderBook()
orderbook.add_order(order_id=1, price=100.5, quantity=100, side="buy")
trades = orderbook.match_orders()
```

## Architecture & Optimization

**Hardware-Level Optimizations:**
- SIMD vectorization for parallel processing
- 64-byte cache line alignment
- Memory prefetching strategies
- NUMA-aware thread pinning

**Software-Level Optimizations:**
- Lock-free data structures with atomic operations
- Zero-copy design in critical paths
- Memory pooling for order objects
- Inline functions and template metaprogramming

**Core Principles:**
1. Latency First - every microsecond matters
2. Cache Awareness - optimize for L1/L2/L3
3. Lock-Free Algorithms - minimize synchronization
4. Predictable Performance - avoid hot path allocations

## Integration with Other Modules

This module integrates with:
- **strategy/**: Provides execution backend for strategies
- **realtime_trading/**: Powers live trading execution
- **risk_control/**: Enforces risk limits during execution
- **scripts/**: Used by evaluation and analysis scripts

## Testing & Troubleshooting

**Run Tests:**
```bash
pytest execution/tests/              # Unit tests
python benchmarks/benchmark_throughput.py  # Performance tests
python test_build.py                # Build verification
```

**Clean Build:**
```bash
cd cpp_core && rm -rf build/
python setup.py clean --all
python setup.py build_ext --inplace
```

**Performance Debugging:**
- CPU frequency: `cat /proc/cpuinfo | grep MHz`
- NUMA config: `numactl --hardware`
- Cache misses: `perf stat -e cache-misses ./program`

## Contributing

1. Measure first - use benchmarks to identify bottlenecks
2. Profile with `perf`, `valgrind`, or `Intel VTune`
3. Verify correctness and performance impact
4. Document non-obvious optimizations

## Further Reading

- [Main System README](../README.md)
- [Strategy Module](../strategy/README.md)
- [Real-time Trading](../realtime_trading/README.md)

---

*This module represents the core of what makes HFT possible: ultra-low-latency execution through hardware-efficient design, lock-free algorithms, and optimized memory management.*

# CUDA-Accelerated Backtesting & Risk Analytics

GPU-accelerated parallel backtesting and Monte Carlo simulation for quantitative trading.

## Overview

This module provides CUDA kernels for compute-intensive operations in quantitative finance:

- **Parallel Backtesting**: Test 1000+ strategies simultaneously (50x CPU speedup)
- **Monte Carlo Risk**: VaR/CVaR with millions of paths (250x CPU speedup)
- **Option Pricing**: Batch pricing and Greeks calculation (167x CPU speedup)
- **Stress Testing**: Multi-scenario portfolio analysis

**Key Principle**: Use CUDA for batch computation, not real-time execution.

## Performance Targets

| Component | CPU (64 cores) | GPU (RTX 4090) | Speedup |
|:----------|:---------------|:---------------|:--------|
| 1000 strategy backtest | 25 min | 30 sec | 50x |
| Monte Carlo (10M paths) | 5 sec | 20 ms | 250x |
| Portfolio VaR | 1.2 sec | 5 ms | 240x |
| Option Greeks (1000) | 500 ms | 3 ms | 167x |

## Prerequisites

- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- CUDA Toolkit 11.8+ or 12.x
- CMake 3.18+
- Python 3.10+
- CuPy (pip install cupy-cuda11x or cupy-cuda12x)

## Installation

### Quick Install

```bash
cd cuda_accelerated
./build_cuda.sh --install
```

### Manual Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
```

### Python-Only Mode

If you don't have CUDA development tools, the Python API will use CuPy-based implementations:

```bash
pip install cupy-cuda11x numpy pandas
```

## Quick Start

### Example 1: Parallel Backtesting

```python
from cuda_backtest import CUDABacktestEngine, MarketData, create_momentum_strategies
import pandas as pd

# Load market data
df = pd.read_csv('SPY_2023.csv', parse_dates=['date'], index_col='date')
market_data = MarketData.from_dataframe(df)

# Create 760 momentum strategies (38 lookbacks x 20 thresholds)
strategies = create_momentum_strategies(
    lookback_range=(10, 200),
    lookback_step=5,
    threshold_range=(0.01, 0.10),
    threshold_points=20
)

# Run parallel backtest on GPU
engine = CUDABacktestEngine(initial_capital=100_000)
results = engine.run(strategies, market_data)

# Print results
print(results.summary())

# Get best strategy
best = results.get_best_strategy(metric='sharpe')
print(f"Best Sharpe: {best.sharpe_ratio:.2f}")
print(f"Return: {best.total_return:.2%}")
```

### Example 2: Portfolio VaR/CVaR

```python
from cuda_monte_carlo import CUDAMonteCarloEngine
import numpy as np

# Initialize with 1 million paths
mc = CUDAMonteCarloEngine(num_paths=1_000_000)

# Define portfolio
portfolio = {
    'SPY': {'position': 1000, 'price': 450, 'drift': 0.08, 'vol': 0.18},
    'TLT': {'position': 500, 'price': 95, 'drift': 0.03, 'vol': 0.12},
    'GLD': {'position': 200, 'price': 180, 'drift': 0.05, 'vol': 0.15}
}

# Correlation matrix
correlation = np.array([
    [1.0, -0.3, 0.2],
    [-0.3, 1.0, -0.1],
    [0.2, -0.1, 1.0]
])

# Calculate 1-month VaR
results = mc.simulate_portfolio(portfolio, time_horizon=1/12, correlation=correlation)

print(f"VaR (95%): ${results.calculate_var(0.95):,.2f}")
print(f"CVaR (95%): ${results.calculate_cvar(0.95):,.2f}")
print(f"Computation time: {results.computation_time:.3f}s")
```

### Example 3: Stress Testing

```python
from cuda_monte_carlo import CUDAMonteCarloEngine

mc = CUDAMonteCarloEngine(num_paths=100_000)

scenarios = [
    {'name': '2008 Crisis', 'drift': -0.35, 'volatility': 0.45},
    {'name': 'COVID Crash', 'drift': -0.30, 'volatility': 0.60},
    {'name': 'Normal Market', 'drift': 0.10, 'volatility': 0.18}
]

results_df = mc.stress_test(portfolio_value=1_000_000, scenarios=scenarios)
print(results_df)
```

## Architecture

```
cuda_accelerated/
├── kernels/                      # CUDA kernel implementations
│   ├── backtest_kernel.cu       # Parallel backtesting
│   ├── monte_carlo_kernel.cu    # Path generation & pricing
│   └── analytics_kernel.cu      # Performance metrics
│
├── host/                         # C++ host code (optional)
│   ├── backtest_engine.cpp
│   └── monte_carlo_engine.cpp
│
├── python/                       # Python interface
│   ├── cuda_backtest.py         # High-level backtest API
│   └── cuda_monte_carlo.py      # Risk analytics API
│
├── bindings/                     # Pybind11 bindings (optional)
│   ├── backtest_bindings.cpp
│   └── monte_carlo_bindings.cpp
│
└── build_cuda.sh                 # Build script
```

## When to Use CUDA vs C++

### Use CUDA (Batch Operations)

- Parameter sweep backtesting (100+ combinations)
- Monte Carlo simulation (10K+ paths)
- Portfolio optimization (multi-asset)
- Rolling analytics (1000+ windows)
- Scenario analysis (stress testing)

### Use C++ (Low Latency)

- Real-time order execution (<1μs)
- Single strategy backtest (quick iteration)
- Live risk checks (per-order)
- WebSocket data handling
- Order book updates

## Advanced Usage

Custom strategies and batch option pricing - see Python API documentation for details.

## Troubleshooting

**CUDA not available**: Check `nvidia-smi` and `nvcc --version`, reinstall CuPy if needed.

**Out of memory**: Reduce num_paths or batch size.

**Slow performance**: First run compiles kernels. Verify GPU usage with `nvidia-smi`.

## Integration with HFT System

CUDA for overnight backtesting and risk calculation. C++ for real-time execution (<1μs).
Workflow: CUDA finds best params → Deploy to C++ engine → Live trading with risk limits.

## References

- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/
- CuPy Documentation: https://docs.cupy.dev/
- Pybind11: https://pybind11.readthedocs.io/

## License

MIT License - see LICENSE file for details

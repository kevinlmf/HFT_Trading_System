# Slippage Performance Benchmark

## Overview

In high-frequency trading, the speed of the slippage calculation directly dictates order latency. This benchmark compares four implementations—Python loop, NumPy vectorised Python, C++, and CUDA—to quantify the throughput gap.

## Features

- **Multiple backends** – measure loop-based Python, vectorised Python, C++, and CUDA side by side.  
- **Batch testing** – run thousands of orders per batch to mimic production workloads.  
- **Performance metrics** – collect execution time, orders-per-second, and speed-up ratios.  
- **Visual report** – export comparison plots for quick reviews.

## Usage

### 1. Install dependencies

```bash
# C++ extension (pybind11 bindings)
cd HFT_System/Execution/cpp_core
pip install -e .

# Optional CUDA path (requires CUDA toolkit + GPU)
pip install cupy
```

### 2. Run the benchmark

```bash
cd HFT_System/Monitoring/benchmarks
python benchmark_slippage.py
```

### 3. Inspect results

- Performance summary is printed to the terminal.  
- Charts are written to `slippage_benchmark_results.png`.

## Expected Performance

Typical uplift across implementations:

1. **Python loop** – baseline; slowest variant.  
2. **NumPy vectorised** – 5–10x faster than the loop.  
3. **C++ (pybind11)** – 10–50x faster than the loop.  
4. **CUDA** – 50–200x faster for large batches (dependent on GPU/PCIe bandwidth).

## Implementation Notes

### Python
- `calculate_slippage_python()` – reference loop implementation.  
- `calculate_slippage_python_vectorized()` – NumPy vectorised path.

### C++
- `SlippageCalculator` in `Execution/cpp_core/include/slippage_calculator.hpp`.  
- Wrapped for Python via pybind11.

### CUDA
- Kernel: `cuda_accelerated/kernels/slippage_kernel.cu`.  
- Python wrapper: `cuda_accelerated/python/cuda_slippage.py`.  
- Supports either native CUDA builds or CuPy kernels.

## HFT Scenario Guidance

Assuming ~1,000 orders per second:

- **Python loop** – unlikely to meet latency targets.  
- **Vectorised Python** – workable for prototyping, still high latency.  
- **C++** – production-ready for CPU-only deployments.  
- **CUDA** – best option for massive burst capacity and sub-millisecond tails.

## Notes & Caveats

1. **C++ module** – rebuild `Execution/cpp_core` when updating headers.  
2. **CUDA module** – requires a CUDA-capable GPU and matching drivers.  
3. **Synthetic data** – benchmark feeds random price/quantity samples; real-world performance depends on data shape.

## Directory Layout

```
Monitoring/benchmarks/
├── benchmark_slippage.py        # Benchmark driver
└── README_SLIPPAGE_BENCHMARK.md # This document

Execution/cpp_core/
├── include/slippage_calculator.hpp
└── src/slippage_calculator.cpp  # Thin wrapper, implementation in header

cuda_accelerated/
├── kernels/slippage_kernel.cu
└── python/cuda_slippage.py
```

## Planned Improvements

- [ ] Add alternative slippage models (e.g., Almgren–Chriss).  
- [ ] Multi-GPU scaling tests.  
- [ ] Memory-usage profiling.  
- [ ] Streaming benchmark for live market data.
# High-Frequency Trading System (HFT)

A unified research-to-execution stack for quantitative trading. The system fuses EDA, intelligent execution (Python/C++/CUDA), multi-strategy evaluation, risk control, position management, and deep optimization into a single automated flow.

---
## Integrated Flow Feedback Loop
1. **Latency Core (engineering)** – `pipeline` + `smart_executor` auto-route workloads across Python vectorized, C++, or CUDA paths; benchmarks live under `benchmark_slippage.py`.
2. **Strategy Evaluation Loop** – `strategy_factory` builds candidates; `strategy_benchmark` + `strategy_analyzer` run backtest, Monte Carlo, and regime-aware scoring, then export results via `result_generator`.
3. **Optimization Stack (5 layers)** – `optimization_stack.py` applies objectives (Sharpe/Variance/CVaR), defines risk models, auto-selects algorithms, optimizes data layout, and executes with JIT/GPU to produce signals & portfolio weights.

Running `./run_trading.sh complete-flow` ties the loop together: data is processed with the latency core, strategies are stress-tested, and the optimization stack finalizes risk metrics, allocations, and reports. Generated artifacts in `results/strategies/` feed back into research and configuration, enabling continuous improvement.

## Why Speed Matters

**Latency Core (sub-millisecond execution)**
- Python vectorized path for fast prototyping.
- Pybind11 C++ engine (<1 µs per order, 100M+ orders/sec throughput).
- CUDA kernels for slippage, Monte Carlo, and backtest acceleration (up to 200M orders/sec).
- Smart executor auto-routes workloads by data size, aligning with cache/JIT/GPU availability.

## **Trading Intelligence (full-stack decision support)**

- Strategy coverage: classical, statistical, ML, RL, and HFT microstructure playbooks.
- Backtesting + Monte Carlo fusion with regime-aware scoring.
- VaR/CVaR, drawdown, volatility, and Sharpe/Sortino metrics with optimized portfolio allocation.
- Automated reporting (JSON/CSV/HTML/PNG) plus textual desk recommendations.

---
## Usage of Statistics Computing
# Optimization Stack (5 Layers)
1. **Statistical Theory** – objectives (Sharpe, variance, CVaR, target-return).
2. **Model Expression** – likelihood/risk functions with gradients/constraints.
3. **Algorithm Design** – automatic choice among GD, Adam, L-BFGS, Newton, annealing.
4. **Data Structure** – SIMD-friendly layouts, parallel chunking, GPU memory plans.
5. **System Implementation** – Numba JIT, CuPy/CUDA kernels, multi-thread execution.

---

## Quick Start
```bash
# 0. Clone and enter the project
git clone https://github.com/kevinlmf/HFT_Trading_System.git hft_system
 cd hft_system/HFT_System

# 1. Make helper scripts executable (first time only)
HFT_System$ chmod +x build_system.sh run_trading.sh

# 2. (Optional) Build C++/CUDA components and verify dependencies
HFT_System$ ./build_system.sh --all --test

# 3. Run the complete trading flow (EDA → strategy evaluation → risk/positions → reports)
HFT_System$ ./run_trading.sh complete-flow --symbols AAPL,MSFT,GOOGL

# 4. Other entrypoints
HFT_System$ ./run_trading.sh paper --dashboard        # Paper trading + dashboard
HFT_System$ ./run_trading.sh backtest --dashboard     # Backtest + dashboard
HFT_System$ ./run_trading.sh benchmark-slippage       # Python/C++/CUDA latency benchmark
```
> Tip: When copying commands into the terminal, drop the trailing comments that start with `#`.
Requirements: Python 3.8+, GCC 7+, optional CUDA 11+. Install dependencies with `pip install -r requirements.txt` (see repository for curated list).

---

## Latency Benchmarks
Monitoring/benchmarks/benchmark_slippage.py` runs these comparisons end-to-end and exports charts under `results/latency/`.

### Statistics Computing Benchmark
Compare full pipeline runtime with the optimization stack enabled vs disabled:
```bash
python3 Monitoring/benchmarks/benchmark_statistics_computing.py
```
Results land in `results/strategies/statistics_benchmark/` as timestamped JSON/TXT summaries (runtime, top strategies, speedup).

**Latest snapshot (2025‑11‑09 14:21:05)**  
- Records: 50,000 (`--records 50000`) with 200,000 Monte Carlo paths (`--monte-carlo-paths 200000`)  
- Baseline runtime: 452.53 s | Optimized runtime: 447.53 s → **1.01× speedup** on CPU-optimized risk metrics  
- Risk checks: `hft_market_making` passed; other strategies flagged for negative Sharpe during stress run  
- Best overall (baseline): `hft_market_making`; best overall (optimized): `momentum`  
- Outputs: `statistics_computing_comparison_20251109_142105.json` and `.txt`, plus per-strategy breakdowns

---

## Primary CLI Entrypoints
- `./run_trading.sh complete-flow` – full pipeline + reports (default strategies or user selection).
- `./run_trading.sh paper` – paper trading engine with monitoring dashboard.
- `./run_trading.sh backtest` – historical evaluation with visual dashboards.
- `./run_trading.sh benchmark-slippage` – compare Python vs C++ vs CUDA slippage implementations.

All modes accept flags: `--symbols`, `--capital`, `--risk-model`, `--strategies`, `--monte-carlo-paths`, etc. Use `./run_trading.sh --help` for full list.

---

## Outputs
Organised under `results/` (auto-ignored by git):
- `strategies/` – contains `trading_analysis_<ts>.*` bundles (JSON, CSV, HTML, PNG, TXT) from complete-flow runs.
- `latency/` – contains `slippage_benchmark_<ts>.*` outputs comparing Python vs C++ vs CUDA.

### Latest Complete-Flow Snapshot (2025‑11‑09 12:58)
- Mode: `./run_trading.sh complete-flow --symbols AAPL,MSFT,GOOGL`
- Risk checks passed: `mean_reversion`, `statistical_mean_reversion`
- Recommended strategy: `mean_reversion` (Sharpe 7.088, total return 5.19e11 %)
- Portfolio allocation: `AAPL` 33.33 %, `MSFT` 33.33 %, `GOOGL` 33.33 %
- Reports saved under `results/trading_analysis_20251109_125803.*`

---

## Key Modules & Scripts
- `Execution/engine/pipeline.py` – orchestrates EDA and smart execution.
- `Execution/engine/integrated_trading_flow.py` – main public interface (invoked by CLI).
- `Execution/engine/strategy_factory.py` – builds classical, ML, RL, HFT strategies.
- `Execution/engine/strategy_analyzer.py` – regime-aware performance diagnostics.
- `Execution/strategy_comparison/strategy_benchmark.py` – backtest + Monte Carlo fusion.
- `Execution/risk_control/portfolio_manager.py` – constraint-aware portfolio optimization.
- `Monitoring/benchmarks/benchmark_slippage.py` – timing harness for Python/C++/CUDA.
- `examples/complete_flow_demo.py` – callable demo of the integrated pipeline.

---
# Safety & Licensing
**Use for research and education. Live deployment requires rigorous validation.** Trading carries risk; past performance is not predictive.

License: MIT

---

When curiosity meets motion, every millisecond holds the spark of alpha. 🌅

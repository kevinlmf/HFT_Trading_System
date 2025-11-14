# System Architecture

## Overview

The HFT Trading System follows a data-driven, multi-layered architecture that integrates market microstructure analysis, strategy selection, validation, and execution into a unified pipeline.

## System Architecture Diagram

```
Market Data Connectors
    ↓
    ├─→ QDB (Quantitative Database) ─→ [Data Storage & Versioning]
    │
    └─→ Market Microstructure Analysis (Research Framework)
            ↓
        Strategy Selection (based on microstructure insights)
            ↓
        Quick Validation (Backtest + Monte Carlo)
            ↓
        Smart Execution → Strategy Evaluation → Optimization Stack → Risk Control → Portfolio Management → Monitoring & Reporting
```

## 11-Layer Pipeline (Complete Flow)

| **Layer** | **Function** | **Output** |
|:----------|:-------------|:-----------|
| **1. Data Connectors** | `Data/connectors/` connect to multiple data sources | Raw market data streams |
| **2. QDB (Quantitative Database)** | `Data/qdb/` stores data with O(log n) indexing, LRU caching, versioning | Standardized, cached, versioned market data |
| **3. Market Microstructure Analysis** | `Research/` analyzes data: profiling → factor hypothesis → validation | Market regime insights, validated factors |
| **4. Strategy Selection** | Based on microstructure analysis, select optimal strategy | Strategy instance optimized for current market |
| **5. Quick Validation** | Lightweight backtest + simplified Monte Carlo validation (< 100ms) | Validated strategies ready for execution |
| **6. Smart Execution** | `Execution/engine/smart_executor.py` selects Python/C++/CUDA paths | Latency-optimized transformations |
| **7. Strategy Evaluation** | `Execution/strategy_comparison/strategy_benchmark.py` performs backtests, Monte Carlo | Risk metrics, performance summaries |
| **8. Optimization Stack** | `Optimization/optimization_stack.py` tunes objectives, constraints, signals (optional, for complete-flow mode) | Optimized weights, signals, risk metrics |
| **9. Risk Control** | `Execution/risk_control/portfolio_manager.py` enforces limits | Risk-adjusted positions |
| **10. Portfolio Management** | Position modules compute allocations & execute orders | Executable orders, allocation plan |
| **11. Monitoring & Reporting** | `Execution/reporting/result_generator.py` produces dashboards | JSON/CSV/HTML/PNG reports, latency benchmarks, final P&L, Sharpe ratio |



### Trading Flow (Real-Time Mode)

**Step 1: Data Ingestion**
- `Data/connectors/` connect to multiple data sources (Alpaca, Binance, Polygon, Yahoo, etc.)
- Data streams flow simultaneously to:
  - **QDB RealtimeCollector**: Automatically stores data to QDB database
  - **Trading Engine**: Collects data in memory for real-time analysis

**Step 2: Data Collection & Storage**
- QDB automatically collects and stores market data with:
  - O(log n) indexing for fast retrieval
  - LRU caching (500x speedup)
  - Data versioning for reproducibility
- Trading Engine collects initial data (default: 100 ticks per symbol)

**Step 3: Market Microstructure Analysis**
- `Research/` framework analyzes collected data:
  - Market microstructure profiling (volatility, liquidity, spread analysis)
  - Factor hypothesis generation (economics-motivated factors)
  - Statistical validation (regression, portfolio tests)
  - ML validation (feature importance, model validation)

**Step 4: Optimal Strategy Selection**
- Based on microstructure analysis results:
  - High volatility → Mean reversion strategies preferred
  - Low volatility → Momentum strategies preferred
  - Liquidity analysis → Position sizing adjustments
- Strategy router selects optimal strategy for current market regime

**Step 5: Quick Validation (Backtest + Monte Carlo)**
- **Lightweight backtest**: Validates strategy on recent 50 ticks (< 100ms)
- **Simplified Monte Carlo**: 1000 paths simulation for risk assessment (< 50ms)
- **Caching**: Results cached for 60s to avoid repeated calculations (< 1ms cache hits)
- **Timeout protection**: 500ms max validation time, fail-safe design
- Only validated strategies proceed to execution

**Step 6: Execution & Monitoring**
- Execute validated trades with risk controls
- Real-time P&L tracking and performance metrics
- Performance monitoring: tick latency, signal generation latency, validation latency
- Final results: Net P&L, Sharpe ratio, total runtime, throughput metrics




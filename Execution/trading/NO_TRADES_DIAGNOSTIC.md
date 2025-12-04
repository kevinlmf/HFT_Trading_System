# Why No Trades Are Executing - Diagnostic Guide

## Common Reasons for Zero Trades

### 1. Confidence Thresholds Too High

The system has multiple confidence filters that may be blocking trades:

- **Strategy Router Confidence**: Must be >= 0.2
- **Signal Confidence**: Must be >= 0.25
- **Signal Strength**: Must be > 0.05

If any of these thresholds are not met, the signal is rejected.

### 2. Insufficient Data

The system requires:
- **Minimum 10 data points** before generating signals
- Strategy-specific lookback windows (usually 20+ periods)

If data collection is slow or insufficient, signals won't be generated.

### 3. Strategy Validation Failed

If `enable_validation=True`, signals go through quick backtest/Monte Carlo validation:
- **Timeout**: 500ms maximum
- **Minimum ticks**: 50 for validation

If validation fails or times out, the trade is skipped.

### 4. Market Regime Detection

The adaptive router may not detect a clear market regime, leading to low confidence in strategy selection.

## Solutions

### Option 1: Lower Thresholds (Quick Fix)

Modify `Execution/trading/trading_engine.py`:

```python
# Line 463: Lower strategy router confidence
if confidence < 0.1:  # Changed from 0.2

# Line 588: Lower signal confidence
if confidence < 0.15:  # Changed from 0.25

# Line 594/597: Lower signal strength
if action == 'BUY' and strength > 0.01:  # Changed from 0.05
```

### Option 2: Disable Validation Temporarily

Set `enable_validation=False` when creating the engine:

```python
engine = RealTimeTradingEngine(
    ...
    enable_validation=False,  # Disable validation
    ...
)
```

### Option 3: Increase Update Frequency

Collect data faster:

```python
update_interval=1.0  # Check every 1 second instead of 5
```

### Option 4: Use Paper Trading Mode with Faster Data

Use a connector with real-time data (Alpaca) instead of delayed data (Yahoo Finance).

## Quick Diagnostic Script

Run this to check what's blocking trades:

```python
# Check confidence thresholds
# Check data collection status
# Check strategy signal generation
```


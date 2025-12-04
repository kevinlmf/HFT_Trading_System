# Trading Strategies Guide

This document describes all available trading strategies in the HFT system.

## Strategy Categories

### 1. Traditional Strategies (2 strategies)

#### Momentum Strategy
- **Name**: `momentum`
- **Description**: Trend-following strategy that buys when price has been rising and sells when falling
- **Method**: Uses rolling return windows to detect momentum
- **Use Case**: Works well in trending markets

#### Mean Reversion Strategy
- **Name**: `mean_reversion`
- **Description**: Counter-trend strategy that trades against price deviations
- **Method**: Uses Z-score to detect when price deviates from moving average
- **Use Case**: Works well in ranging/volatile markets

---

### 2. Machine Learning Strategies (2 strategies)

#### Random Forest Strategy
- **Name**: `ml_random_forest`
- **Description**: Ensemble learning strategy using Random Forest classifier
- **Features**: 
  - Multiple lookback windows (5, 10, 20 periods)
  - Rolling statistics (mean, std)
  - Price-to-moving-average ratios
- **Method**: Predicts future return direction using ensemble of decision trees
- **Use Case**: Captures non-linear patterns in market data

#### XGBoost Strategy
- **Name**: `ml_xgboost`
- **Description**: Gradient boosting strategy using XGBoost
- **Features**:
  - Multiple lookback windows (5, 10, 20, 30 periods)
  - Rolling statistics
  - Volume indicators (if available)
- **Method**: Uses gradient boosting with probability thresholds for signal generation
- **Requirements**: XGBoost library (`pip install xgboost`)
- **Use Case**: High-performance ML predictions with feature importance

---

### 3. Reinforcement Learning Strategies (1 strategy)

#### Simple RL Strategy
- **Name**: `rl_simple`
- **Description**: Simplified reinforcement learning approach using state-based decision making
- **Features**:
  - State space: Rolling means and stds at multiple timeframes
  - Action space: Buy, Sell, Hold
- **Method**: State-value function approximation with threshold-based decisions
- **Use Case**: Adaptive strategy that learns from market states
- **Note**: This is a simplified implementation. Full RL requires training environment.

---

### 4. LLM-Based Strategies (2 strategies)

#### LLM Sentiment Strategy
- **Name**: `llm_sentiment`
- **Description**: Strategy based on pattern recognition (simulated LLM analysis)
- **Features**:
  - Trend detection
  - Volatility analysis
  - Pattern recognition
- **Method**: Analyzes price patterns and market conditions to generate signals
- **Use Case**: Can be enhanced with real LLM APIs (GPT-4, Claude) for news/ sentiment analysis

#### LLM Pattern Strategy
- **Name**: `llm_pattern`
- **Description**: Technical pattern recognition strategy
- **Features**:
  - Peak/trough detection
  - Chart pattern identification
  - Trend analysis
- **Method**: Identifies technical patterns like head-and-shoulders, double tops/bottoms
- **Use Case**: Pattern-based trading signals

---

## Usage

All strategies are automatically included when running:

```bash
./run_trading.sh complete-flow --symbols AAPL,MSFT
```

The system will:
1. Test all available strategies
2. Calculate performance metrics for each
3. Compare and rank strategies
4. Save results to `results/strategies/`

## Strategy Comparison

The system automatically compares all strategies on:
- **Total Return**: Overall profit/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Volatility**: Return standard deviation
- **Max Drawdown**: Maximum loss from peak
- **HFT Metrics**: Hit ratio, latency, slippage, etc.

## Adding New Strategies

To add a new strategy:

1. Create a function that takes `data: pd.DataFrame` and returns `pd.Series` (signals)
2. Add it to `create_sample_strategies()` in `Execution/engine/integrated_trading_flow.py`
3. Strategy should handle:
   - Missing data gracefully
   - Return signals as -1 (sell), 0 (hold), 1 (buy)
   - Use the same index as input data

Example:

```python
def my_custom_strategy(data: pd.DataFrame) -> pd.Series:
    """My custom strategy description"""
    prices = data['close']
    # Your strategy logic here
    signals = ...  # Your signals
    return signals.fillna(0)
```

## Performance Expectations

- **Traditional**: Fast, interpretable, moderate performance
- **ML**: Higher performance potential, requires more data, slower
- **RL**: Adaptive, requires training, can be unstable
- **LLM**: Pattern recognition, can be enhanced with real LLM APIs

## Dependencies

- **Base**: pandas, numpy (always required)
- **ML**: scikit-learn (for Random Forest), xgboost (optional, for XGBoost)
- **RL**: PyTorch (optional, for full RL implementation)
- **LLM**: None (current implementation is pattern-based, can integrate OpenAI/Anthropic APIs)

## Future Enhancements

- Full RL implementation with training environment
- Real LLM API integration (GPT-4, Claude) for news sentiment
- Deep learning strategies (LSTM, Transformer)
- Ensemble strategies combining multiple methods


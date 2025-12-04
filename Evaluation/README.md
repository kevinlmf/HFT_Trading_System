# 07. Evaluation - 评估

## 功能

对策略的表现进行量化回测或实盘评估。

**产出**: evaluation reports & diagnostics

## 评估指标

- **PnL Decomposition** - 盈亏分解
- **Sharpe / Sortino** - 风险调整收益
- **Win Rate** - 胜率
- **Alpha Decay** - Alpha衰减
- **Execution Slippage** - 执行滑点

## 使用示例

```python
from 07_Evaluation import StrategyEvaluator

evaluator = StrategyEvaluator()
results = evaluator.evaluate(strategy, backtest_data)
print(f"Sharpe: {results.sharpe_ratio}")
```


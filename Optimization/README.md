# 08. Optimization - 模型/策略优化

## 功能

根据评估结果调整系统。

**产出**: optimized strategy parameters

## 优化内容

- **策略参数优化** - Strategy parameter tuning
- **执行调优** - Execution tuning
- **超参数搜索** - Hyperparameter search
- **奖励塑形** - Reward shaping (RL)

## 使用示例

```python
from 08_Optimization import OptimizationStack

optimizer = OptimizationStack()
optimized_params = optimizer.optimize(
    strategy, evaluation_results, objective="sharpe"
)
```


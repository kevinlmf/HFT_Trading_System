# 05. Strategy Construction - 策略生成

## 功能

把 alpha 转成具体的交易操作或仓位。

**产出**: trading decisions（方向、大小、开平仓点）

## 策略类型

- **Mean Reversion** - 均值回归
- **Momentum** - 动量策略
- **Threshold Rules** - 阈值规则
- **RL Policy** - 强化学习策略
- **Market Making** - 做市策略

## 使用示例

```python
from 05_Strategy_Construction import MomentumStrategy

strategy = MomentumStrategy(lookback=20, threshold=0.02)
decisions = strategy.generate_signals(alpha_signals, market_data)
```


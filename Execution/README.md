# 06. Execution - 智能执行

## 功能

把策略意图转化成真实可成交订单。

**产出**: executed orders（含 slippage & impact）

## 执行算法

- **TWAP** - Time-Weighted Average Price
- **VWAP** - Volume-Weighted Average Price
- **Adaptive Execution** - 自适应执行
- **Smart Routing** - 智能路由
- **Slippage Minimization** - 滑点最小化

## 使用示例

```python
from 06_Execution.engine.smart_executor import SmartExecutor

executor = SmartExecutor()
orders, info = executor.execute_slippage_calculation(
    prices, quantities, mid_prices, sides
)
```


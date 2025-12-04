# 09. Risk Control - 风险控制

## 功能

确保策略不会爆仓或失控。

**产出**: 风险约束与实时保护

## 风险控制措施

- **Exposure Limit** - 敞口限制
- **Volatility Targeting** - 波动率目标
- **CVaR / ES** - 条件风险价值
- **Copula-based Tail Risk** - 尾部风险
- **Stop-loss / Kill-switch** - 止损/紧急停止

## 使用示例

```python
from 09_Risk_Control import RiskManager

risk_manager = RiskManager(
    max_exposure=100000,
    max_drawdown=0.20,
    stop_loss=0.05
)
approved_orders = risk_manager.check_orders(orders, current_positions)
```


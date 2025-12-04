# 04. Alpha Modeling - 信号构建

## 功能

把 microstructure → 转换成 可预测未来方向或风险的 alpha。

**产出**: alpha_t（预测信号）

## 支持的模型

- **统计模型**: AR/VAR/Kalman
- **机器学习**: LSTM/RF
- **强化学习**: RL value/policy signals
- **风险模型**: Copula risk signals

## 使用示例

```python
from 04_Alpha_Modeling import FactorHypothesisGenerator, MLValidator

# 生成因子假设
generator = FactorHypothesisGenerator()
factors = generator.generate(microstructure_signals)

# 验证模型
validator = MLValidator()
alpha = validator.validate(factors, returns)
```


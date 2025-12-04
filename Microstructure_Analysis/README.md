# 03. Microstructure Analysis - 微观结构分析

## 功能

从 LOB 与逐笔数据中提取结构规律与特征。

**产出**: microstructure signals

## 分析内容

- **Order Flow Imbalance** - 订单流不平衡
- **Queue Dynamics** - 队列动态
- **Spread 和成交密度** - 价差与流动性
- **Short-term Mean Reversion** - 短期均值回归

## 使用示例

```python
from 03_Microstructure_Analysis import MicrostructureProfiler

profiler = MicrostructureProfiler()
metrics = profiler.analyze(lob_data, trades_data)
```


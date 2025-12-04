# 10. Monitoring - 实时监控

## 功能

实时监测系统运行状态。

**产出**: 实时 dashboard + Alerts

## 监控指标

- **Latency** - 延迟
- **PnL & Drawdown** - 盈亏与回撤
- **Slippage** - 滑点
- **Signal Health** - 信号健康度
- **Risk Exposure** - 风险敞口

## 使用示例

```python
from 10_Monitoring import RealtimeMonitor

monitor = RealtimeMonitor()
monitor.track_pnl(positions, prices)
monitor.track_latency(execution_times)
alerts = monitor.check_thresholds()
```


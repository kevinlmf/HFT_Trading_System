# HFT 指标优化指南

## 快速开始

### 启用优化
```bash
# 运行时启用HFT优化
ENABLE_HFT_OPTIMIZATION=true ./run_trading.sh complete-flow --symbols AAPL,MSFT
```

## 优化策略详解

### 1. Hit Ratio 优化 (目标: >55%)

#### 当前问题
- momentum: 48.27%
- mean_reversion: 50.89%

#### 优化方法

**a) 信号过滤**
```python
# 添加置信度阈值
min_confidence = 0.6  # 只保留高置信度信号
signal_strength = abs(signal) / volatility
filtered_signals = signals[signal_strength > min_confidence]
```

**b) 多时间框架确认**
```python
# 使用多个移动平均线确认趋势
short_ma = prices.rolling(5).mean()
medium_ma = prices.rolling(20).mean()
long_ma = prices.rolling(50).mean()

# 只在趋势一致时交易
trend_aligned = (short_ma > medium_ma > long_ma) or (short_ma < medium_ma < long_ma)
```

**c) 成交量确认**
```python
# 只在成交量放大时交易
volume_ma = volume.rolling(20).mean()
high_volume = volume > volume_ma * 1.2
signals = signals[high_volume]
```

**d) 减少假信号**
```python
# 使用中值滤波减少噪声
filtered = signals.rolling(3, center=True).median()
# 只保留显著变化
significant = abs(signals.diff()) > threshold
```

#### 预期效果
- 提升 5-10% Hit Ratio
- 从 48-51% → 53-61%

### 2. Latency Jitter 优化 (目标: <2ms)

#### 当前问题
- 计算异常（已修复）
- 需要实际降低延迟波动

#### 优化方法

**a) 向量化计算**
```python
# 使用NumPy向量化而非循环
# 慢: for i in range(n): signals[i] = calculate(i)
# 快: signals = np.where(condition, 1, -1)
```

**b) 预计算指标**
```python
# 预计算常用指标，避免重复计算
precomputed = {
    'returns': prices.pct_change(),
    'volatility': returns.rolling(20).std(),
    'ma': prices.rolling(20).mean()
}
```

**c) 批量处理**
```python
# 批量处理而非逐个处理
batch_size = 100
for i in range(0, n, batch_size):
    process_batch(data[i:i+batch_size])
```

**d) 减少数据复制**
```python
# 使用视图而非复制
view = data.iloc[start:end]  # 视图
copy = data.iloc[start:end].copy()  # 复制（慢）
```

#### 预期效果
- 降低 50-70% 延迟波动
- 从异常值 → <2ms

### 3. Throughput 优化 (目标: 1000-10000 TPS)

#### 当前问题
- 0 TPS（计算异常，已修复）
- 需要提高实际处理能力

#### 优化方法

**a) 结果缓存**
```python
# 缓存计算结果
cache = {}
def strategy(data):
    key = hash(data.index[-10:])
    if key in cache:
        return cache[key]
    result = compute(data)
    cache[key] = result
    return result
```

**b) 增量计算**
```python
# 只计算新数据，复用旧结果
def incremental_strategy(new_data, previous_result):
    # 只处理新数据
    new_signals = compute(new_data[-10:])
    # 合并结果
    return pd.concat([previous_result, new_signals])
```

**c) 并行处理**
```python
from multiprocessing import Pool

def parallel_strategy(data_chunks):
    with Pool() as pool:
        results = pool.map(compute_strategy, data_chunks)
    return pd.concat(results)
```

**d) 减少I/O操作**
```python
# 批量读取而非逐行读取
# 慢: for row in data: process(row)
# 快: process_batch(data)
```

#### 预期效果
- 提升 10-100倍吞吐量
- 从 0 TPS → 1000-10000 TPS

### 4. Slippage 优化 (目标: <1 bps)

#### 当前表现
- momentum: 0.04 bps ✅ (优秀)
- mean_reversion: 0.03 bps ✅ (优秀)

#### 保持优势的方法

**a) 使用限价单**
```python
# 使用限价单而非市价单
if signal > 0:
    order = LimitOrder(price=bid_price, side=BUY)
else:
    order = LimitOrder(price=ask_price, side=SELL)
```

**b) 订单拆分**
```python
# 大单拆分执行
large_order = 10000
chunks = [2500, 2500, 2500, 2500]  # 分成4份
for chunk in chunks:
    execute(chunk)
```

**c) 选择最佳执行时机**
```python
# 在spread较小时执行
spread = ask_price - bid_price
if spread < threshold:
    execute_order()
else:
    wait_for_better_price()
```

**d) 考虑订单簿深度**
```python
# 检查订单簿深度
depth = sum(order_book['bid_size'][:5])
if depth > min_depth:
    execute_order()
```

## 综合优化流程

### 步骤1: 评估当前性能
```bash
./run_trading.sh complete-flow --symbols AAPL,MSFT
# 查看 results/hft_metrics/ 中的指标
```

### 步骤2: 应用优化
```bash
ENABLE_HFT_OPTIMIZATION=true ./run_trading.sh complete-flow --symbols AAPL,MSFT
```

### 步骤3: 对比结果
```bash
# 对比优化前后的指标
# - Hit Ratio 应该提升
# - Latency Jitter 应该降低
# - Throughput 应该提高
```

## 高级优化技巧

### 1. 机器学习增强
```python
# 使用ML模型预测信号质量
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
features = [returns, volatility, volume, ...]
signal_quality = model.predict_proba(features)
high_quality_signals = signals[signal_quality > 0.7]
```

### 2. 实时自适应
```python
# 根据市场条件动态调整参数
if volatility > threshold:
    min_confidence = 0.7  # 提高阈值
else:
    min_confidence = 0.6
```

### 3. A/B测试
```python
# 对比不同优化策略
strategies = {
    'baseline': original_strategy,
    'optimized': optimized_strategy
}
results = compare_strategies(strategies, data)
```

## 监控和调优

### 关键指标监控
1. **Hit Ratio**: 每日监控，目标 >55%
2. **Latency Jitter**: 实时监控，目标 <2ms
3. **Throughput**: 峰值监控，目标 >1000 TPS
4. **Slippage**: 每笔交易监控，目标 <1 bps

### 调优建议
- 每周回顾一次指标
- 根据市场条件调整参数
- 持续A/B测试新策略
- 记录优化历史，分析效果

## 预期优化效果

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| Hit Ratio | 48-51% | 53-61% | +5-10% |
| Latency Jitter | 异常 | <2ms | -50-70% |
| Throughput | 0 TPS | 1000-10000 TPS | +1000%+ |
| Slippage | 0.03-0.04 bps | <0.03 bps | 保持优秀 |


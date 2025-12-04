# 02. QDB - Quant Database

## 功能

负责清洗、标准化、对齐和版本管理。

**产出**: 结构化 quant-ready 数据（features, bars, LOB snapshots）

## 核心特性

- **数据标准化**: 统一的数据Schema
- **O(log n) 索引**: 快速数据检索
- **LRU缓存**: 500x 加速
- **版本管理**: 实验可复现性
- **实时收集**: 自动数据摄取

## 使用示例

```python
from 02_QDB import create_qdb

qdb = create_qdb(base_path="./data/qdb")
qdb.store("AAPL", df, data_version="v1.0")
data = qdb.load("AAPL", start="2024-01-01", end="2024-12-31")
```

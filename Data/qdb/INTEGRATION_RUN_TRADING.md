# QDB集成到run_trading.sh说明

## ✅ 集成完成

QDB系统已成功集成到`run_trading.sh`脚本中，所有交易模式现在都支持QDB作为统一数据源。

## 🎯 集成特性

### 1. 自动QDB初始化
- 脚本启动时自动初始化QDB
- 如果QDB模块不存在，会优雅降级到传统模式
- 显示QDB状态和配置信息

### 2. 实时数据收集（Paper/Live模式）
- 使用`RealtimeCollector`自动收集实时数据
- 数据自动存储到QDB，带版本标签
- 缓冲区批量写入，提高性能

### 3. 历史数据加载（Backtest模式）
- 从QDB加载历史数据进行回测
- 确保回测数据一致性
- 支持版本选择

### 4. 完整流程集成（Complete-flow模式）
- 所有策略使用QDB数据
- 版本追踪每个实验
- 数据一致性保证

## 📋 新增命令行选项

### QDB相关选项

```bash
--qdb-path PATH        # QDB数据存储路径（默认: ./Data/datasets/qdb）
--no-qdb               # 禁用QDB集成（使用传统数据加载）
--qdb-version VERSION  # QDB数据版本标签（默认: qdb_YYYYMMDD）
```

## 🚀 使用示例

### 1. Paper Trading（默认启用QDB）
```bash
./run_trading.sh paper --symbols AAPL,MSFT,GOOGL
```

### 2. Paper Trading（自定义QDB版本）
```bash
./run_trading.sh paper --symbols AAPL --qdb-version qdb_2024Q1
```

### 3. Paper Trading（禁用QDB，使用传统模式）
```bash
./run_trading.sh paper --no-qdb --symbols AAPL
```

### 4. Live Trading（启用QDB实时收集）
```bash
./run_trading.sh live --capital 50000 --qdb-version qdb_live_2024
```

### 5. Backtest（使用QDB数据）
```bash
./run_trading.sh backtest --symbols AAPL,MSFT --qdb-path ./Data/datasets/qdb
```

### 6. Complete Flow（QDB集成）
```bash
./run_trading.sh complete-flow --symbols AAPL,MSFT --qdb-path ./Data/qdb
```

## 🔧 配置说明

### 默认配置
- `ENABLE_QDB=true` - 默认启用QDB
- `QDB_PATH="./Data/datasets/qdb"` - 默认存储路径
- `QDB_DATA_VERSION="qdb_$(date +%Y%m%d)"` - 自动生成版本标签
- `QDB_BUFFER_SIZE=1000` - 实时收集缓冲区大小

### 环境变量
脚本会自动导出以下环境变量供Python代码使用：
- `ENABLE_QDB` - 是否启用QDB
- `QDB_PATH` - QDB存储路径
- `QDB_DATA_VERSION` - 数据版本标签
- `QDB_BUFFER_SIZE` - 缓冲区大小

## 📊 工作流程

### Paper/Live Trading流程
1. 初始化QDB
2. 初始化数据连接器（Alpaca等）
3. 创建`RealtimeCollector`（如果QDB启用）
4. 启动实时数据收集
5. 运行交易引擎
6. 数据自动存储到QDB
7. 停止时刷新缓冲区

### Backtest流程
1. 初始化QDB
2. 从QDB加载历史数据
3. 运行回测
4. 结果保存（可选：保存到QDB）

### Complete Flow流程
1. 初始化QDB
2. 从QDB加载数据（或生成测试数据）
3. 运行完整流程（EDA + Strategy + Risk）
4. 所有数据使用QDB版本管理

## 🎁 优势

### 1. 数据一致性
- 所有模式（实盘/回测/模拟）都从同一QDB数据源取数据
- 避免"策略在回测中赚钱，实盘中亏钱"的问题

### 2. 可复现性
- 每个实验都记录数据版本
- 可以完全复现任何历史实验

### 3. 性能提升
- Parquet列式存储，快速加载（<10ms目标）
- LRU缓存，多策略并行训练时共享数据
- 缓冲区批量写入，减少IO

### 4. 自动化
- 实时数据自动收集和存储
- 无需手动管理数据文件
- 版本自动追踪

## ⚠️ 注意事项

1. **QDB模块可选**: 如果QDB模块不存在，脚本会优雅降级到传统模式
2. **向后兼容**: 使用`--no-qdb`可以禁用QDB，使用传统数据加载方式
3. **存储空间**: QDB使用Parquet格式，压缩率高，但仍需注意磁盘空间
4. **版本管理**: 建议为不同实验使用不同的版本标签

## 🔍 调试

### 查看QDB状态
脚本启动时会显示：
```
✓ QDB initialized successfully
  Storage path: ./Data/datasets/qdb
  Data version: qdb_20241109
```

### 查看QDB数据
```python
from Data.qdb import create_qdb

qdb = create_qdb()
print(qdb.list_symbols())  # 列出所有symbol
print(qdb.get_info())      # 获取QDB信息
```

### 查看缓存统计
```python
stats = qdb.get_cache_stats()
print(f"命中率: {stats['hit_rate']*100:.2f}%")
```

## 📝 总结

QDB已完全集成到`run_trading.sh`中，现在：
- ✅ 所有交易模式都支持QDB
- ✅ 实时数据自动收集和存储
- ✅ 历史数据统一管理
- ✅ 版本追踪和实验复现
- ✅ 向后兼容（可禁用QDB）

系统现在有了统一的数据基础设施！


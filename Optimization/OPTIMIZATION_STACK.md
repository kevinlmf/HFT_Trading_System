# 深度优化栈文档

## 概述

深度优化栈实现了从统计理论到系统实现的完整优化流程，包含5个层次：

```
[统计理论层]     ➜ 定义模型目标
       ↓
[模型表达层]     ➜ 写出似然/风险函数
       ↓
[算法设计层]     ➜ 选择求解算法（优化、采样、近似）
       ↓
[数据结构层]     ➜ 优化内存布局、并行结构
       ↓
[系统实现层]     ➜ 编译、并行、GPU、0/1信号执行
```

## 架构层次

### 1. 统计理论层 (Statistical Theory Layer)

**功能**: 定义模型目标和约束

**主要类**:
- `StatisticalTheoryLayer`: 统计理论层
- `StatisticalModel`: 统计模型定义
- `ModelObjective`: 模型目标枚举

**支持的优化目标**:
- `MAXIMIZE_SHARPE`: 最大化Sharpe比率
- `MINIMIZE_VARIANCE`: 最小化方差
- `MAXIMIZE_RETURN`: 最大化收益
- `MINIMIZE_CVAR`: 最小化CVaR
- `MAXIMIZE_UTILITY`: 最大化效用函数
- `MINIMIZE_TRACKING_ERROR`: 最小化跟踪误差

**使用示例**:
```python
from Optimization.optimization_stack import StatisticalTheoryLayer, ModelObjective

layer = StatisticalTheoryLayer()
model = layer.define_model(
    name="portfolio_opt",
    objective=ModelObjective.MAXIMIZE_SHARPE,
    constraints={'max_position_size': 0.2}
)
```

### 2. 模型表达层 (Model Expression Layer)

**功能**: 定义似然函数和风险函数

**主要类**:
- `ModelExpressionLayer`: 模型表达层
- `LikelihoodFunction`: 似然函数
- `RiskFunction`: 风险函数

**预定义的函数**:
- `gaussian_likelihood()`: 高斯似然函数
- `variance_risk()`: 方差风险函数
- `cvar_risk()`: CVaR风险函数

**使用示例**:
```python
from Optimization.optimization_stack import ModelExpressionLayer
import numpy as np

layer = ModelExpressionLayer()
returns = np.random.randn(1000, 10) * 0.02

# 定义风险函数
risk_func = layer.variance_risk(returns)

# 自定义风险函数
def custom_risk(weights):
    return np.dot(weights, np.dot(cov_matrix, weights))

risk_func = layer.define_risk_function("custom", custom_risk)
```

### 3. 算法设计层 (Algorithm Design Layer)

**功能**: 根据问题特征自动选择最优算法

**主要类**:
- `AlgorithmDesignLayer`: 算法设计层
- `AlgorithmConfig`: 算法配置
- `AlgorithmType`: 算法类型枚举

**支持的算法**:
- `GRADIENT_DESCENT`: 梯度下降
- `NEWTON_METHOD`: 牛顿法
- `QUASI_NEWTON`: 拟牛顿法
- `STOCHASTIC_GRADIENT`: 随机梯度下降
- `ADAM`: Adam优化器
- `LBFGS`: L-BFGS算法
- `SIMULATED_ANNEALING`: 模拟退火
- `GENETIC_ALGORITHM`: 遗传算法
- `MCMC`: 马尔可夫链蒙特卡洛
- `VARIATIONAL_INFERENCE`: 变分推断

**算法选择逻辑**:
- 有Hessian且问题规模<1000 → 牛顿法
- 有梯度且问题规模>10000 → 随机梯度下降（GPU）
- 有梯度且问题规模>1000 → Adam
- 有梯度且问题规模<1000 → L-BFGS
- 无梯度 → 模拟退火

**使用示例**:
```python
from Optimization.optimization_stack import AlgorithmDesignLayer

layer = AlgorithmDesignLayer()
config = layer.select_algorithm(
    problem_type='optimization',
    has_gradient=True,
    problem_size=100,
    gpu_available=False
)

# 执行优化
optimal_params, info = layer.optimize(
    objective_func,
    initial_params,
    config,
    gradient_func=gradient_func
)
```

### 4. 数据结构层 (Data Structure Layer)

**功能**: 优化内存布局和并行结构

**主要类**:
- `DataStructureLayer`: 数据结构层
- `DataStructure`: 数据结构定义
- `MemoryLayout`: 内存布局枚举

**内存布局类型**:
- `ROW_MAJOR`: 行主序（C-style）
- `COLUMN_MAJOR`: 列主序（Fortran-style）
- `BLOCKED`: 分块存储
- `SPARSE`: 稀疏矩阵

**优化特性**:
- 内存对齐（SIMD优化）
- 缓存感知布局
- 并行数据结构

**使用示例**:
```python
from Optimization.optimization_stack import DataStructureLayer
import numpy as np

layer = DataStructureLayer()
data = np.random.randn(1000, 10)

# 优化内存布局
structure, optimized_data = layer.optimize_layout(
    data,
    access_pattern="row",
    parallel=True
)

# 创建并行结构
data, chunks = layer.create_parallel_structure(data, num_threads=4)
```

### 5. 系统实现层 (System Implementation Layer)

**功能**: 编译、并行、GPU执行、信号生成

**主要类**:
- `SystemImplementationLayer`: 系统实现层
- `SystemConfig`: 系统配置
- `ExecutionBackend`: 执行后端枚举

**执行后端**:
- `CPU`: CPU执行
- `GPU`: GPU执行（CUDA）
- `FPGA`: FPGA执行
- `ASIC`: ASIC执行

**功能**:
- JIT编译（Numba）
- GPU加速（CuPy）
- 并行执行（多线程/多进程）
- 0/1信号生成

**使用示例**:
```python
from Optimization.optimization_stack import SystemImplementationLayer, ExecutionBackend

layer = SystemImplementationLayer()

# 编译函数
compiled_func = layer.compile_function(my_func, config)

# GPU执行
if layer.gpu_available:
    result = layer.execute_on_gpu(func, data)

# 并行执行
result = layer.execute_parallel(func, data, num_threads=4)

# 生成交易信号
signals = layer.generate_signal(weights, threshold=0.0)
```

## 完整使用示例

### 投资组合优化

```python
from Optimization.optimization_stack import OptimizationStack, ModelObjective
import numpy as np

# 创建优化栈
stack = OptimizationStack()

# 准备数据
returns = np.random.randn(1000, 10) * 0.02

# 执行优化
result = stack.optimize_portfolio(
    returns,
    objective=ModelObjective.MAXIMIZE_SHARPE,
    constraints={'max_position_size': 0.2, 'sum_to_one': True}
)

print(f"最优权重: {result['optimal_weights']}")
print(f"交易信号: {result['signals']}")
print(f"使用算法: {result['algorithm']}")
```

### 集成到交易流程

优化栈已自动集成到 `IntegratedTradingFlow` 中：

```python
from Execution.engine.integrated_trading_flow import IntegratedTradingFlow

# 创建流程（自动使用优化栈）
flow = IntegratedTradingFlow(
    initial_capital=1000000,
    use_optimization_stack=True  # 默认启用
)

# 执行流程
result = flow.execute_complete_flow_with_position_management(...)
```

## 性能优化

### 1. 编译优化
- 使用Numba JIT编译加速数值计算
- 支持CPU和GPU编译

### 2. 并行优化
- 自动检测问题规模
- 选择合适的并行策略
- 支持多线程和多进程

### 3. GPU加速
- 自动检测GPU可用性
- 大规模问题自动使用GPU
- 支持CuPy加速

### 4. 内存优化
- 内存对齐（SIMD）
- 缓存感知布局
- 减少内存拷贝

## 依赖要求

### 必需
- NumPy
- SciPy
- Pandas

### 可选（性能提升）
- Numba: JIT编译加速
- CuPy: GPU加速
- scikit-learn: ML算法支持

安装可选依赖：
```bash
pip install numba cupy scikit-learn
```

## 最佳实践

1. **问题规模**: 
   - 小规模（<100）: 使用精确算法（牛顿法）
   - 中规模（100-1000）: 使用L-BFGS
   - 大规模（>1000）: 使用随机梯度下降或Adam

2. **GPU使用**:
   - 问题规模>1000时自动启用
   - 需要安装CuPy

3. **内存优化**:
   - 根据访问模式选择内存布局
   - 行访问用ROW_MAJOR，列访问用COLUMN_MAJOR

4. **算法选择**:
   - 有梯度时优先使用梯度算法
   - 无梯度时使用启发式算法

## 扩展开发

### 添加新的优化目标

```python
from Optimization.optimization_stack import ModelObjective

# 在ModelObjective枚举中添加新目标
class ModelObjective(Enum):
    # ... 现有目标
    MAXIMIZE_SORTINO = "maximize_sortino"
```

### 添加新的算法

```python
from Optimization.optimization_stack import AlgorithmType, AlgorithmDesignLayer

# 在AlgorithmType中添加新算法
class AlgorithmType(Enum):
    # ... 现有算法
    ADAGRAD = "adagrad"

# 在AlgorithmDesignLayer中实现
def _adagrad_optimizer(self, ...):
    # 实现Adagrad算法
    pass
```

### 添加新的执行后端

```python
from Optimization.optimization_stack import ExecutionBackend

# 在ExecutionBackend中添加新后端
class ExecutionBackend(Enum):
    # ... 现有后端
    TPU = "tpu"
```

## 性能基准

在标准测试（10资产，1000样本）上：

- **CPU优化**: 比原始实现快 2-5x
- **GPU加速**: 比CPU快 10-50x（大规模问题）
- **编译优化**: 比Python快 5-20x

## 故障排除

### Numba编译错误
- 使用 `forceobj=True` 允许Python对象
- 或禁用编译，使用纯Python

### GPU不可用
- 检查CuPy安装
- 检查CUDA驱动
- 系统会自动回退到CPU

### 内存不足
- 减少问题规模
- 使用稀疏矩阵
- 分批处理








"""
深度优化栈模块
从统计理论到系统实现的完整优化流程

优化版本（集成QDB和优化方法/数据结构）：
- EnhancedOptimizationStack: 增强的优化栈
- EnhancedOptimizedPortfolioManager: 增强的投资组合管理器
"""
from .optimization_stack import (
    OptimizationStack,
    StatisticalTheoryLayer,
    ModelExpressionLayer,
    AlgorithmDesignLayer,
    DataStructureLayer,
    SystemImplementationLayer,
    ModelObjective,
    AlgorithmType,
    ExecutionBackend
)

# 优化版本（集成QDB）
try:
    from .optimized_optimization_stack import (
        EnhancedOptimizationStack,
        OptimizedDataLoader,
        OptimizedAlgorithmSelector,
        OptimizedDataStructure
    )
    from .enhanced_portfolio_manager import EnhancedOptimizedPortfolioManager
    OPTIMIZED_VERSION_AVAILABLE = True
except ImportError:
    OPTIMIZED_VERSION_AVAILABLE = False
    EnhancedOptimizationStack = None
    OptimizedDataLoader = None
    OptimizedAlgorithmSelector = None
    OptimizedDataStructure = None
    EnhancedOptimizedPortfolioManager = None

__all__ = [
    # 标准版本
    'OptimizationStack',
    'StatisticalTheoryLayer',
    'ModelExpressionLayer',
    'AlgorithmDesignLayer',
    'DataStructureLayer',
    'SystemImplementationLayer',
    'ModelObjective',
    'AlgorithmType',
    'ExecutionBackend',
]

# 如果优化版本可用，添加到导出列表
if OPTIMIZED_VERSION_AVAILABLE:
    __all__.extend([
        'EnhancedOptimizationStack',
        'OptimizedDataLoader',
        'OptimizedAlgorithmSelector',
        'OptimizedDataStructure',
        'EnhancedOptimizedPortfolioManager',
    ])








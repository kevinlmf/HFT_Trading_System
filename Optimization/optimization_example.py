"""
Optimization模块优化使用示例

展示如何使用优化后的Optimization模块（集成QDB和优化方法/数据结构）
"""

import numpy as np
import pandas as pd
from datetime import datetime
from Optimization.optimized_optimization_stack import (
    EnhancedOptimizationStack,
    OptimizedDataLoader
)
from Optimization.optimization_stack import ModelObjective
from Optimization.enhanced_portfolio_manager import EnhancedOptimizedPortfolioManager
from Execution.risk_control.portfolio_manager import RiskModel


def example_qdb_integrated_optimization():
    """示例：使用QDB集成的优化栈"""
    print("=" * 80)
    print("示例1: QDB集成的投资组合优化")
    print("=" * 80)
    
    # 创建增强的优化栈（集成QDB）
    stack = EnhancedOptimizationStack(use_qdb=True)
    
    # 从QDB加载数据并优化（快速，O(log n)）
    print("\n从QDB加载数据并优化...")
    result = stack.optimize_portfolio_from_qdb(
        symbols=['SPY', 'AAPL', 'MSFT', 'GOOGL'],
        start_time='2024-01-01',
        end_time='2024-12-31',
        objective=ModelObjective.MAXIMIZE_SHARPE
    )
    
    print(f"\n优化结果:")
    print(f"  最优权重: {result['optimal_weights']}")
    print(f"  使用算法: {result['optimization_info']['algorithm']}")
    print(f"  数据源: {result['optimization_info']['data_source']}")
    print(f"  缓存使用: {result['optimization_info']['cache_used']}")
    print(f"  Sharpe比率: {result['performance_metrics']['sharpe_ratio']:.4f}")
    
    # 第二次优化（相同数据，使用缓存）
    print("\n第二次优化（使用缓存）...")
    result2 = stack.optimize_portfolio_from_qdb(
        symbols=['SPY', 'AAPL', 'MSFT', 'GOOGL'],
        start_time='2024-01-01',
        end_time='2024-12-31',
        objective=ModelObjective.MAXIMIZE_SHARPE
    )
    print(f"  缓存命中: {result2['optimization_info']['cache_used']}")


def example_enhanced_portfolio_manager():
    """示例：使用增强的投资组合管理器"""
    print("\n" + "=" * 80)
    print("示例2: 增强的投资组合管理器")
    print("=" * 80)
    
    # 创建增强的管理器
    manager = EnhancedOptimizedPortfolioManager(
        initial_capital=1000000.0,
        risk_model=RiskModel.MAXIMIZE_SHARPE,
        use_qdb=True
    )
    
    # 从QDB加载数据并优化
    print("\n计算最优权重...")
    weights = manager.calculate_optimal_weights(
        symbols=['SPY', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        start_time='2024-01-01',
        end_time='2024-12-31',
        use_qdb=True
    )
    
    print(f"\n最优权重:")
    for symbol, weight in weights.items():
        print(f"  {symbol}: {weight:.4f}")
    
    # 查看优化信息
    info = manager.get_optimization_info()
    print(f"\n优化信息:")
    print(f"  QDB启用: {info['qdb_enabled']}")
    print(f"  GPU可用: {info['gpu_available']}")
    print(f"  缓存大小: {info['cache_size']}")


def example_method_optimization():
    """示例：方法优化效果"""
    print("\n" + "=" * 80)
    print("示例3: 方法优化效果对比")
    print("=" * 80)
    
    import time
    
    # 创建测试数据
    n_samples, n_assets = 1000, 50
    returns = np.random.randn(n_samples, n_assets).astype(np.float32) * 0.02
    
    # 测试协方差矩阵计算（无缓存）
    print("\n1. 协方差矩阵计算（无缓存）:")
    start = time.time()
    cov1 = np.cov(returns, rowvar=False)
    time1 = (time.time() - start) * 1000
    print(f"   时间: {time1:.2f}ms")
    
    # 测试协方差矩阵计算（有缓存）
    data_loader = OptimizedDataLoader(use_qdb=False)
    print("\n2. 协方差矩阵计算（有缓存，第一次）:")
    start = time.time()
    cov2 = data_loader.get_covariance_matrix(returns, use_cache=True, cache_key="test")
    time2 = (time.time() - start) * 1000
    print(f"   时间: {time2:.2f}ms")
    
    print("\n3. 协方差矩阵计算（有缓存，第二次）:")
    start = time.time()
    cov3 = data_loader.get_covariance_matrix(returns, use_cache=True, cache_key="test")
    time3 = (time.time() - start) * 1000
    print(f"   时间: {time3:.2f}ms (缓存命中)")
    print(f"   加速比: {time1/time3:.0f}x" if time3 > 0 else "  加速比: N/A")
    
    # 测试数据结构优化
    print("\n4. 数据结构优化（float32 vs float64）:")
    returns_f64 = returns.astype(np.float64)
    returns_f32 = returns.astype(np.float32)
    
    size_f64 = returns_f64.nbytes / 1024
    size_f32 = returns_f32.nbytes / 1024
    
    print(f"   float64大小: {size_f64:.2f}KB")
    print(f"   float32大小: {size_f32:.2f}KB")
    print(f"   内存节省: {(1 - size_f32/size_f64)*100:.1f}%")


def example_algorithm_selection():
    """示例：智能算法选择"""
    print("\n" + "=" * 80)
    print("示例4: 智能算法选择")
    print("=" * 80)
    
    from Optimization.optimized_optimization_stack import OptimizedAlgorithmSelector
    
    selector = OptimizedAlgorithmSelector()
    
    test_cases = [
        (10, True, False, None, False, False),  # 小规模
        (100, True, False, None, False, False),  # 中等规模
        (500, True, False, None, False, False),  # 大规模
        (1000, True, False, None, False, True),  # 超大规模 + GPU
    ]
    
    print("\n算法选择结果:")
    print(f"{'问题规模':<10} {'有梯度':<8} {'GPU':<8} {'选择算法':<25}")
    print("-" * 60)
    
    for size, has_grad, has_hess, cond_num, sparse, gpu in test_cases:
        algo = selector.select_optimal_algorithm(
            problem_size=size,
            has_gradient=has_grad,
            has_hessian=has_hess,
            condition_number=cond_num,
            is_sparse=sparse,
            gpu_available=gpu
        )
        print(f"{size:<10} {str(has_grad):<8} {str(gpu):<8} {algo.value:<25}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Optimization模块优化示例")
    print("=" * 80)
    
    # 运行示例
    try:
        example_qdb_integrated_optimization()
    except Exception as e:
        print(f"\n⚠️  示例1失败（可能需要QDB数据）: {e}")
    
    try:
        example_enhanced_portfolio_manager()
    except Exception as e:
        print(f"\n⚠️  示例2失败（可能需要QDB数据）: {e}")
    
    example_method_optimization()
    example_algorithm_selection()
    
    print("\n" + "=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)














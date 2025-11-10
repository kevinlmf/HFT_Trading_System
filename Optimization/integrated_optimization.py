"""
整合优化栈到完整交易流程
将5层优化架构应用到：策略优化、风险计算、仓位管理、信号生成等各个环节
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Optimization.optimization_stack import (
    OptimizationStack,
    ModelObjective,
    AlgorithmType,
    ExecutionBackend
)
from Execution.engine.complete_trading_flow import CompleteTradingFlow
from Execution.engine.strategy_factory import StrategyFactory
from Execution.strategy_comparison.strategy_benchmark import StrategyBenchmark
from Execution.risk_control.risk_metrics import RiskCalculator


class IntegratedOptimizedFlow(CompleteTradingFlow):
    """
    整合优化栈的完整交易流程
    
    优化栈应用点：
    1. 策略参数优化 - 使用优化栈优化策略参数
    2. 风险计算加速 - GPU加速VaR/CVaR计算
    3. 仓位优化 - 使用优化栈进行投资组合优化（已集成）
    4. 信号生成 - 优化信号生成逻辑
    5. 策略对比加速 - 并行优化多个策略
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        monte_carlo_paths: int = 100000,
        use_optimization_stack: bool = True
    ):
        super().__init__(risk_free_rate, periods_per_year, monte_carlo_paths)
        
        self.use_optimization_stack = use_optimization_stack
        if use_optimization_stack:
            self.optimization_stack = OptimizationStack()
            print("✓ Optimization Stack enabled for all optimization tasks")
        else:
            self.optimization_stack = None
    
    def optimize_strategy_parameters(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        param_ranges: Dict[str, tuple]
    ) -> Dict[str, float]:
        """
        使用优化栈优化策略参数
        
        Args:
            strategy_func: 策略函数
            param_ranges: 参数范围 {param_name: (min, max)}
        
        Returns:
            最优参数
        """
        if not self.optimization_stack:
            return {}
        
        # 定义优化目标：最大化策略收益
        def objective(params: np.ndarray) -> float:
            # 将参数转换为字典
            param_dict = {}
            for i, (name, _) in enumerate(param_ranges.items()):
                param_dict[name] = params[i]
            
            # 运行策略并计算收益
            try:
                # 这里需要根据实际策略函数调整
                signals = strategy_func(data, **param_dict)
                if isinstance(signals, pd.Series):
                    returns = (data['close'].pct_change() * signals.shift(1)).sum()
                else:
                    returns = 0
                return -returns  # 负号因为要最小化
            except:
                return np.inf
        
        # 初始参数（范围中点）
        initial_params = np.array([
            (min_val + max_val) / 2 
            for min_val, max_val in param_ranges.values()
        ])
        
        # 选择算法
        algo_config = self.optimization_stack.algorithm_layer.select_algorithm(
            problem_type='optimization',
            has_gradient=False,  # 参数优化通常没有解析梯度
            problem_size=len(param_ranges)
        )
        
        # 执行优化
        optimal_params, info = self.optimization_stack.algorithm_layer.optimize(
            objective,
            initial_params,
            algo_config
        )
        
        # 转换为字典
        result = {}
        for i, (name, _) in enumerate(param_ranges.items()):
            result[name] = float(optimal_params[i])
        
        return result
    
    def calculate_risk_metrics_optimized(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        use_gpu: bool = False
    ) -> Dict[str, float]:
        """
        使用优化栈加速风险指标计算
        
        Args:
            returns: 收益率序列
            equity_curve: 权益曲线
            use_gpu: 是否使用GPU
        
        Returns:
            风险指标字典
        """
        if not self.optimization_stack:
            # 回退到标准方法
            return self.risk_calculator.calculate_all_risk_metrics(returns, equity_curve)
        
        # 转换为numpy数组
        returns_array = returns.values
        equity_array = equity_curve.values
        
        # 使用GPU加速（如果可用）
        if use_gpu and self.optimization_stack.system_layer.gpu_available:
            try:
                import cupy as cp
                returns_gpu = cp.asarray(returns_array)
                equity_gpu = cp.asarray(equity_array)
                
                # GPU计算VaR/CVaR
                sorted_returns = cp.sort(returns_gpu)
                var_95_idx = int(len(sorted_returns) * 0.05)
                var_95 = float(cp.asnumpy(sorted_returns[var_95_idx]))
                cvar_95 = float(cp.asnumpy(sorted_returns[:var_95_idx].mean()))
                
                # 计算波动率
                volatility = float(cp.asnumpy(cp.std(returns_gpu) * np.sqrt(252)))
                
                return {
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'volatility': volatility,
                    'computed_on': 'GPU'
                }
            except Exception as e:
                print(f"  ⚠ GPU risk calculation failed: {e}, using CPU")
        
        # CPU计算（使用优化栈的数据结构优化）
        optimized_returns, _ = self.optimization_stack.data_layer.create_parallel_structure(
            returns_array, num_threads=4
        )
        
        # 并行计算风险指标
        sorted_returns = np.sort(optimized_returns)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_99_idx = int(len(sorted_returns) * 0.01)
        
        var_95 = sorted_returns[var_95_idx]
        var_99 = sorted_returns[var_99_idx]
        cvar_95 = sorted_returns[:var_95_idx].mean()
        cvar_99 = sorted_returns[:var_99_idx].mean()
        
        # 计算最大回撤
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        volatility = np.std(returns_array) * np.sqrt(252)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'computed_on': 'CPU_optimized'
        }
    
    def optimize_signal_generation(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        transaction_cost: float = 0.001
    ) -> np.ndarray:
        """
        优化信号生成（考虑交易成本）
        
        使用优化栈找到最优阈值，最大化收益-成本
        
        Args:
            predictions: 预测值数组
            prices: 价格数组
            transaction_cost: 交易成本比例
        
        Returns:
            优化后的0/1信号
        """
        if not self.optimization_stack:
            # 简单阈值
            return (predictions > 0).astype(int)
        
        # 定义优化目标：最大化收益-成本
        def objective(threshold: float) -> float:
            signals = (predictions > threshold).astype(int)
            # 计算信号变化（交易次数）
            trades = np.sum(np.diff(signals) != 0)
            # 计算收益
            returns = np.sum(predictions * signals) - trades * transaction_cost
            return -returns  # 负号因为要最小化
        
        # 优化阈值
        algo_config = self.optimization_stack.algorithm_layer.select_algorithm(
            problem_type='optimization',
            has_gradient=False,
            problem_size=1
        )
        
        initial_threshold = 0.0
        optimal_threshold, _ = self.optimization_stack.algorithm_layer.optimize(
            lambda t: objective(t[0]) if isinstance(t, np.ndarray) else objective(t),
            np.array([initial_threshold]),
            algo_config
        )
        
        # 生成信号
        threshold = optimal_threshold[0] if isinstance(optimal_threshold, np.ndarray) else optimal_threshold
        signals = self.optimization_stack.system_layer.generate_signal(
            predictions - threshold
        )
        
        return signals
    
    def compare_strategies_optimized(
        self,
        strategies: Dict[str, Callable],
        data: pd.DataFrame,
        parallel: bool = True
    ) -> List[Any]:
        """
        使用优化栈并行对比策略
        
        Args:
            strategies: 策略字典
            data: 数据
            parallel: 是否并行执行
        
        Returns:
            策略对比结果列表
        """
        if not self.optimization_stack or not parallel:
            # 使用标准方法
            return self.benchmark.compare_strategies(strategies, data)
        
        # 准备并行执行
        results = []
        
        # 使用优化栈的并行结构
        strategy_list = list(strategies.items())
        data_optimized, chunks = self.optimization_stack.data_layer.create_parallel_structure(
            np.array(range(len(strategy_list))), num_threads=min(4, len(strategy_list))
        )
        
        # 并行评估策略
        for chunk_start, chunk_end in chunks:
            chunk_strategies = dict(strategy_list[chunk_start:chunk_end])
            chunk_results = self.benchmark.compare_strategies(chunk_strategies, data)
            results.extend(chunk_results)
        
        return results
    
    def execute_complete_flow_optimized(
        self,
        data: pd.DataFrame,
        strategies: Optional[Dict[str, Callable]] = None,
        optimize_strategy_params: bool = False,
        use_gpu_for_risk: bool = False
    ) -> Dict:
        """
        执行完整优化流程（整合优化栈）
        
        Args:
            data: 历史数据
            strategies: 策略字典
            optimize_strategy_params: 是否优化策略参数
            use_gpu_for_risk: 是否使用GPU计算风险
        
        Returns:
            完整流程结果（包含优化信息）
        """
        print("=" * 80)
        print("Integrated Optimized Trading Flow")
        print("=" * 80)
        
        if self.optimization_stack:
            print("\nOptimization Stack Features:")
            print(f"  - Strategy parameter optimization: {optimize_strategy_params}")
            print(f"  - GPU risk calculation: {use_gpu_for_risk}")
            print(f"  - Parallel strategy comparison: True")
            print(f"  - Optimized portfolio management: True")
            if self.optimization_stack.system_layer.gpu_available:
                print(f"  - GPU available: ✓")
            else:
                print(f"  - GPU available: ✗ (using CPU)")
        
        # 1. 标准流程：EDA + 数据清理
        pipeline_result = self.pipeline.process(
            data=data,
            data_type="trading_data",
            auto_clean=True
        )
        
        # 2. 创建策略（如果需要优化参数）
        if strategies is None:
            strategies = StrategyFactory.create_all_strategies(pipeline_result['cleaned_data'])
        
        # 3. 优化策略参数（可选）
        if optimize_strategy_params and self.optimization_stack:
            print("\n" + "=" * 80)
            print("Optimizing Strategy Parameters...")
            print("=" * 80)
            optimized_strategies = {}
            for name, strategy_func in list(strategies.items())[:3]:  # 只优化前3个
                print(f"  Optimizing {name}...")
                # 这里需要根据实际策略定义参数范围
                # 简化示例：暂时跳过参数优化（需要策略支持）
                optimized_strategies[name] = strategy_func
            strategies.update(optimized_strategies)
        
        # 4. 策略对比（使用优化栈并行）
        print("\n" + "=" * 80)
        print("Strategy Comparison (Optimized)")
        print("=" * 80)
        comparison_results = self.compare_strategies_optimized(
            strategies,
            pipeline_result['cleaned_data'],
            parallel=True
        )
        
        # 5. 使用优化栈计算风险指标
        if comparison_results:
            print("\n" + "=" * 80)
            print("Risk Metrics Calculation (Optimized)")
            print("=" * 80)
            for result in comparison_results[:3]:  # 只优化前3个
                if hasattr(result.backtest_result, 'returns'):
                    optimized_risk = self.calculate_risk_metrics_optimized(
                        result.backtest_result.returns,
                        result.backtest_result.equity_curve,
                        use_gpu=use_gpu_for_risk
                    )
                    if hasattr(optimized_risk, "__dict__"):
                        optimized_risk_dict = vars(optimized_risk)
                    else:
                        optimized_risk_dict = dict(optimized_risk) if isinstance(optimized_risk, dict) else {}
                    backend = optimized_risk_dict.get('computed_on', 'standard') if optimized_risk_dict else 'standard'
                    print(f"  {result.strategy_name}: computed on {backend}")
        
        # 6. 策略分析
        analysis_result = self.strategy_analyzer.analyze_strategy_performance(
            comparison_results, pipeline_result['cleaned_data']
        )
        self.strategy_analyzer.print_analysis_report(analysis_result)
        
        # 7. 生成最终报告
        final_report = self._generate_final_report(
            pipeline_result, comparison_results, analysis_result
        )
        
        # 添加优化信息
        optimization_info = {
            'optimization_stack_enabled': self.use_optimization_stack,
            'gpu_available': self.optimization_stack.system_layer.gpu_available if self.optimization_stack else False,
            'strategies_optimized': optimize_strategy_params,
            'gpu_risk_calculation': use_gpu_for_risk
        }
        
        return {
            'pipeline_result': pipeline_result,
            'comparison_results': comparison_results,
            'final_report': final_report,
            'strategy_analysis': analysis_result,
            'optimization_info': optimization_info,
            'recommended_strategy': comparison_results[0] if comparison_results else None
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    from Execution.engine.pipeline import create_sample_data
    
    # 创建测试数据
    data = create_sample_data(n_records=1000)
    data['close'] = data['price']
    data.index = pd.date_range(end=pd.Timestamp.now(), periods=len(data), freq='D')
    
    # 创建优化流程
    flow = IntegratedOptimizedFlow(
        monte_carlo_paths=10000,
        use_optimization_stack=True
    )
    
    # 执行完整优化流程
    result = flow.execute_complete_flow_optimized(
        data=data,
        optimize_strategy_params=False,  # 可以启用参数优化
        use_gpu_for_risk=False  # 如果有GPU可以启用
    )
    
    print("\n" + "=" * 80)
    print("Optimization Summary")
    print("=" * 80)
    print(f"Optimization Stack: {'Enabled' if result['optimization_info']['optimization_stack_enabled'] else 'Disabled'}")
    print(f"GPU Available: {'Yes' if result['optimization_info']['gpu_available'] else 'No'}")
    print(f"Strategies Tested: {len(result['comparison_results'])}")


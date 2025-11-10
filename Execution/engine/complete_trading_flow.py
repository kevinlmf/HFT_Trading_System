"""
完整的交易流程
整合：EDA -> 数据清理 -> 智能执行 -> 策略对比 -> 风控 -> 评估
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Execution.engine.pipeline import TradingPipeline
from Execution.engine.strategy_factory import StrategyFactory
from Execution.engine.strategy_analyzer import StrategyAnalyzer
from Execution.strategy_comparison.strategy_benchmark import StrategyBenchmark, StrategyComparisonResult
from Execution.risk_control.risk_metrics import RiskCalculator
from Environment.backtester.simple_backtester import BacktestConfig


class CompleteTradingFlow:
    """
    完整的交易流程
    
    流程：
    1. EDA分析数据
    2. 数据清理
    3. 智能执行（选择Python/C++/CUDA）
    4. 策略对比（Monte Carlo + Backtest）
    5. 风控检查（VaR/CVaR）
    6. Sharpe评估
    7. 综合评分和推荐
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 monte_carlo_paths: int = 100000):
        """
        Args:
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易期数
            monte_carlo_paths: Monte Carlo路径数
        """
        self.pipeline = TradingPipeline()
        self.benchmark = StrategyBenchmark(
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            monte_carlo_paths=monte_carlo_paths
        )
        self.risk_calculator = RiskCalculator(risk_free_rate, periods_per_year)
        self.strategy_analyzer = StrategyAnalyzer()
    
    def execute_complete_flow(self,
                             data: pd.DataFrame,
                             strategies: Optional[Dict[str, Callable]] = None,
                             backtest_config: Optional[BacktestConfig] = None,
                             risk_limits: Optional[Dict[str, float]] = None,
                             use_all_strategies: bool = False,
                             force_slippage_impl: Optional[str] = None) -> Dict:
        """
        执行完整流程
        
        Args:
            data: 历史数据
            strategies: 策略字典 {name: strategy_function}，如果为None且use_all_strategies=True，则使用所有策略
            backtest_config: 回测配置
            risk_limits: 风险限制
            use_all_strategies: 如果为True，自动创建所有可用策略
        
        Returns:
            完整流程结果
        """
        print("="*80)
        print("Complete Trading Flow")
        print("="*80)
        
        # Step 1: EDA + 数据清理 + 智能执行
        print("\n" + "="*80)
        print("Phase 1: Data Processing & Smart Execution")
        print("="*80)
        pipeline_result = self.pipeline.process(
            data=data,
            data_type="trading_data",
            auto_clean=True,
            force_implementation=force_slippage_impl
        )
        
        # 如果没有提供策略或要求使用所有策略，则创建所有策略
        if strategies is None or use_all_strategies:
            print("\n" + "="*80)
            print("Creating All Available Strategies...")
            print("="*80)
            strategies = StrategyFactory.create_all_strategies(pipeline_result['cleaned_data'])
            categories = StrategyFactory.get_strategy_categories()
            print(f"\n策略分类:")
            for category, strategy_list in categories.items():
                available = [s for s in strategy_list if s in strategies]
                if available:
                    print(f"  {category}: {len(available)} strategies")
                    for s in available[:3]:  # 只显示前3个
                        print(f"    - {s}")
                    if len(available) > 3:
                        print(f"    ... and {len(available)-3} more")
            print(f"\n总计: {len(strategies)} 个策略")
        
        # Step 2: 策略对比（Monte Carlo + Backtest）
        print("\n" + "="*80)
        print("Phase 2: Strategy Comparison (Monte Carlo + Backtest)")
        print("="*80)
        comparison_results = self.benchmark.compare_strategies(
            strategies=strategies,
            data=pipeline_result['cleaned_data'],
            backtest_config=backtest_config,
            risk_limits=risk_limits
        )
        
        # Step 3: 策略表现分析（不同市场条件）
        print("\n" + "="*80)
        print("Phase 3: Strategy Performance Analysis")
        print("="*80)
        analysis_result = self.strategy_analyzer.analyze_strategy_performance(
            comparison_results, pipeline_result['cleaned_data']
        )
        self.strategy_analyzer.print_analysis_report(analysis_result)
        
        # Step 4: 综合评估和推荐
        print("\n" + "="*80)
        print("Phase 4: Final Evaluation & Recommendation")
        print("="*80)
        self.benchmark.print_comparison_report(comparison_results)
        
        # 生成最终报告
        final_report = self._generate_final_report(
            pipeline_result, comparison_results, analysis_result
        )
        
        return {
            'pipeline_result': pipeline_result,
            'comparison_results': comparison_results,
            'final_report': final_report,
            'strategy_analysis': analysis_result,
            'recommended_strategy': comparison_results[0] if comparison_results else None
        }
    
    def _generate_final_report(self,
                              pipeline_result: Dict,
                              comparison_results: List[StrategyComparisonResult],
                              analysis_result: Optional[Dict] = None) -> Dict:
        """生成最终报告"""
        report = {
            'data_quality': pipeline_result['profile'].quality.value,
            'data_size': pipeline_result['profile'].n_records,
            'execution_method': pipeline_result['execution_result']['execution_info']['implementation'],
            'n_strategies_tested': len(comparison_results),
            'strategies_passed_risk': sum(1 for r in comparison_results if r.passed_risk_checks),
            'best_strategy': comparison_results[0].strategy_name if comparison_results else None,
            'best_score': comparison_results[0].overall_score if comparison_results else 0.0,
            'best_sharpe': comparison_results[0].sharpe_ratio if comparison_results else 0.0
        }
        
        print("\n" + "="*80)
        print("Final Report")
        print("="*80)
        print(f"Data Quality: {report['data_quality'].upper()}")
        print(f"Data Size: {report['data_size']:,} records")
        print(f"Execution Method: {report['execution_method']}")
        print(f"Strategies Tested: {report['n_strategies_tested']}")
        print(f"Strategies Passed Risk Checks: {report['strategies_passed_risk']}")
        print(f"\nBest Strategy: {report['best_strategy']}")
        print(f"  Score: {report['best_score']:.2f}/100")
        print(f"  Sharpe Ratio: {report['best_sharpe']:.3f}")
        print("="*80)
        
        return report


def create_sample_strategies():
    """创建示例策略用于测试"""
    def momentum_strategy(data: pd.DataFrame):
        """动量策略 - 返回当前信号值（不是Series）"""
        if len(data) < 20:
            return 0
        
        # 简单动量：20日收益率
        if 'close' not in data.columns:
            return 0
        
        returns = data['close'].pct_change(20)
        if len(returns) == 0 or pd.isna(returns.iloc[-1]):
            return 0
        
        current_return = returns.iloc[-1]
        if current_return > 0.02:
            return 1  # 买入
        elif current_return < -0.02:
            return -1  # 卖出
        else:
            return 0  # 持有
    
    def mean_reversion_strategy(data: pd.DataFrame):
        """均值回归策略 - 返回当前信号值"""
        if len(data) < 20:
            return 0
        
        if 'close' not in data.columns:
            return 0
        
        # 简单均值回归：价格偏离20日均线
        ma = data['close'].rolling(20).mean()
        if len(ma) == 0 or pd.isna(ma.iloc[-1]) or ma.iloc[-1] == 0:
            return 0
        
        current_price = data['close'].iloc[-1]
        deviation = (current_price - ma.iloc[-1]) / ma.iloc[-1]
        
        if deviation < -0.05:
            return 1  # 买入（价格低于均值）
        elif deviation > 0.05:
            return -1  # 卖出（价格高于均值）
        else:
            return 0  # 持有
    
    return {
        'momentum': momentum_strategy,
        'mean_reversion': mean_reversion_strategy
    }


if __name__ == "__main__":
    # 示例使用
    from Execution.engine.pipeline import create_sample_data
    
    # 创建测试数据
    data = create_sample_data(n_records=1000)
    data['close'] = data['price']  # 添加close列用于策略
    
    # 创建策略
    strategies = create_sample_strategies()
    
    # 运行完整流程
    flow = CompleteTradingFlow()
    result = flow.execute_complete_flow(
        data=data,
        strategies=strategies
    )
    
    print("\n流程完成！")


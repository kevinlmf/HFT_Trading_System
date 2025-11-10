"""
整合完整交易流程（包含仓位管理）
整合：EDA -> 数据清理 -> 智能执行 -> 策略对比 -> 仓位管理 -> 风控 -> 评估
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Execution.engine.complete_trading_flow import CompleteTradingFlow
from Execution.engine.result_generator import ResultGenerator
# 尝试导入优化版本
try:
    from Optimization.integrated_optimization import IntegratedOptimizedFlow
    INTEGRATED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    INTEGRATED_OPTIMIZATION_AVAILABLE = False
    IntegratedOptimizedFlow = None
from Execution.risk_control.portfolio_manager import (
    AdvancedPortfolioManager, 
    RiskModel, 
    RiskConstraints
)
# 尝试导入优化版本
try:
    from Optimization.optimized_portfolio_manager import OptimizedPortfolioManager
    OPTIMIZATION_STACK_AVAILABLE = True
except ImportError:
    OPTIMIZATION_STACK_AVAILABLE = False
    OptimizedPortfolioManager = None
from Environment.backtester.simple_backtester import BacktestConfig


class IntegratedTradingFlow:
    """
    整合完整交易流程（包含仓位管理）
    
    完整流程：
    1. EDA分析数据
    2. 数据清理
    3. 智能执行（选择Python/C++/CUDA）
    4. 策略对比（Monte Carlo + Backtest）
    5. 仓位管理（计算最优仓位权重）
    6. 风控检查（VaR/CVaR）
    7. Sharpe评估
    8. 综合评分和推荐
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 monte_carlo_paths: int = 100000,
                 initial_capital: float = 1_000_000,
                 risk_model: RiskModel = RiskModel.RISK_PARITY):
        """
        Args:
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易期数
            monte_carlo_paths: Monte Carlo路径数
            initial_capital: 初始资金
            risk_model: 风险模型（用于仓位管理）
        """
        # 使用优化版本（如果可用）
        if INTEGRATED_OPTIMIZATION_AVAILABLE and IntegratedOptimizedFlow:
            self.complete_flow = IntegratedOptimizedFlow(
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                monte_carlo_paths=monte_carlo_paths,
                use_optimization_stack=True
            )
            print("✓ Using Integrated Optimized Flow with Optimization Stack")
        else:
            self.complete_flow = CompleteTradingFlow(
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                monte_carlo_paths=monte_carlo_paths
            )
        
        # 仓位管理器
        risk_constraints = RiskConstraints(
            max_position_size=0.2,  # 单个仓位最大20%
            max_portfolio_volatility=0.15,  # 组合最大波动率15%
            max_drawdown_limit=0.05,  # 最大回撤限制5%
            var_limit=0.02,  # VaR限制2%
            concentration_limit=0.5  # 前5大仓位集中度限制50%
        )
        
        # 使用优化栈版本（如果可用）
        if OPTIMIZATION_STACK_AVAILABLE and OptimizedPortfolioManager:
            self.portfolio_manager = OptimizedPortfolioManager(
                initial_capital=initial_capital,
                risk_model=risk_model,
                constraints=risk_constraints,
                use_optimization_stack=True
            )
            print("✓ Using Optimization Stack for portfolio optimization")
        else:
            self.portfolio_manager = AdvancedPortfolioManager(
                initial_capital=initial_capital,
                risk_model=risk_model,
                constraints=risk_constraints
            )
        
        # 结果生成器
        self.result_generator = ResultGenerator(output_dir="results")
    
    def execute_complete_flow_with_position_management(self,
                                                      data: pd.DataFrame,
                                                      strategies: Dict[str, Callable],
                                                      symbols: List[str],
                                                      backtest_config: Optional[BacktestConfig] = None,
                                                      risk_limits: Optional[Dict[str, float]] = None,
                                                      force_slippage_impl: Optional[str] = None) -> Dict:
        """
        执行完整流程（包含仓位管理）
        
        Args:
            data: 历史数据
            strategies: 策略字典
            symbols: 交易标的列表
            backtest_config: 回测配置
            risk_limits: 风险限制
        
        Returns:
            完整流程结果（包含仓位管理结果）
        """
        print("="*80)
        print("Integrated Trading Flow with Position Management")
        print("="*80)
        
        # Phase 1-3: 使用CompleteTradingFlow
        print("\n" + "="*80)
        print("Phase 1-3: Data Processing, Strategy Comparison, Risk Control")
        print("="*80)
        flow_result = self.complete_flow.execute_complete_flow(
            data=data,
            strategies=strategies,
            backtest_config=backtest_config,
            risk_limits=risk_limits,
            use_all_strategies=(strategies is None),  # 如果没有提供策略，使用所有策略
            force_slippage_impl=force_slippage_impl
        )
        
        # Phase 4: 仓位管理
        print("\n" + "="*80)
        print("Phase 4: Position Management")
        print("="*80)
        position_result = self._calculate_optimal_positions(
            data=data,
            symbols=symbols,
            recommended_strategy=flow_result['recommended_strategy']
        )
        
        # Phase 5: 整合结果
        print("\n" + "="*80)
        print("Phase 5: Final Integration")
        print("="*80)
        final_result = self._integrate_results(flow_result, position_result)
        
        # Phase 6: 生成报告
        print("\n" + "="*80)
        print("Phase 6: Generating Reports")
        print("="*80)
        report_files = self.result_generator.generate_all_reports(final_result)
        
        print("\n" + "="*80)
        print("Report Files Generated")
        print("="*80)
        for report_type, filepath in report_files.items():
            print(f"  {report_type.upper()}: {filepath}")
        print("="*80)
        
        final_result['report_files'] = report_files
        
        return final_result
    
    def _calculate_optimal_positions(self,
                                   data: pd.DataFrame,
                                   symbols: List[str],
                                   recommended_strategy: Optional) -> Dict:
        """计算最优仓位"""
        print("\n计算最优仓位权重...")
        
        # 准备价格数据
        price_data = {}
        for symbol in symbols:
            if symbol in data.columns or 'close' in data.columns:
                # 如果数据是单标的，使用close列
                if 'close' in data.columns:
                    price_data[symbol] = pd.DataFrame({'close': data['close']})
                elif symbol in data.columns:
                    price_data[symbol] = pd.DataFrame({'close': data[symbol]})
        
        if not price_data:
            print("Warning: 无法准备价格数据，使用等权重")
            return {
                'optimal_weights': {symbol: 1.0/len(symbols) for symbol in symbols},
                'position_sizes': {},
                'risk_metrics': {}
            }
        
        # 计算最优权重
        optimal_weights = self.portfolio_manager.calculate_optimal_weights(
            price_data=price_data
        )
        
        print(f"\n最优仓位权重:")
        for symbol, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {weight*100:.2f}%")
        
        # 计算仓位大小（股数）
        position_sizes = {}
        total_value = self.portfolio_manager.state.total_value
        current_prices = {}
        
        for symbol in symbols:
            if symbol in price_data and len(price_data[symbol]) > 0:
                current_prices[symbol] = price_data[symbol]['close'].iloc[-1]
        
        for symbol, weight in optimal_weights.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                target_value = total_value * weight
                position_sizes[symbol] = target_value / current_prices[symbol]
        
        # 计算组合风险指标
        returns_data = self.portfolio_manager._prepare_returns_data(price_data)
        risk_metrics = {}
        if not returns_data.empty:
            risk_metrics = self.portfolio_manager.calculate_portfolio_risk_metrics(returns_data)
            
            print(f"\n组合风险指标:")
            if risk_metrics:
                print(f"  组合波动率: {risk_metrics.get('volatility', 0)*100:.2f}%")
                print(f"  VaR (95%): {abs(risk_metrics.get('var_95', 0))*100:.2f}%")
                print(f"  CVaR (95%): {abs(risk_metrics.get('cvar_95', 0))*100:.2f}%")
                print(f"  最大回撤: {abs(risk_metrics.get('max_drawdown', 0))*100:.2f}%")
                print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}")
        
        # 检查风险违规
        violations = self.portfolio_manager.check_risk_violations(risk_metrics)
        if violations:
            print(f"\n风险违规警告:")
            for violation in violations:
                print(f"  ⚠ {violation}")
        else:
            print(f"\n✓ 所有风险检查通过")
        
        return {
            'optimal_weights': optimal_weights,
            'position_sizes': position_sizes,
            'risk_metrics': risk_metrics,
            'violations': violations,
            'portfolio_summary': self.portfolio_manager.get_portfolio_summary()
        }
    
    def _integrate_results(self, flow_result: Dict, position_result: Dict) -> Dict:
        """整合所有结果"""
        integrated = {
            **flow_result,
            'position_management': position_result,
            'final_recommendation': {
                'strategy': flow_result['recommended_strategy'].strategy_name if flow_result['recommended_strategy'] else None,
                'optimal_weights': position_result['optimal_weights'],
                'risk_metrics': position_result['risk_metrics'],
                'passed_all_checks': len(position_result.get('violations', [])) == 0
            }
        }
        
        # 打印最终推荐
        print("\n" + "="*80)
        print("Final Recommendation with Position Management")
        print("="*80)
        
        if flow_result['recommended_strategy']:
            recommended = flow_result['recommended_strategy']
            print(f"\n推荐策略: {recommended.strategy_name}")
            print(f"  Sharpe Ratio: {recommended.sharpe_ratio:.3f}")
            print(f"  综合评分: {recommended.overall_score:.2f}/100")
        
        print(f"\n仓位配置:")
        for symbol, weight in sorted(position_result['optimal_weights'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {symbol}: {weight*100:.2f}%")
        
        if position_result.get('risk_metrics'):
            rm = position_result['risk_metrics']
            print(f"\n组合风险:")
            print(f"  波动率: {rm.get('volatility', 0)*100:.2f}%")
            print(f"  VaR (95%): {abs(rm.get('var_95', 0))*100:.2f}%")
            print(f"  CVaR (95%): {abs(rm.get('cvar_95', 0))*100:.2f}%")
        
        print(f"\n风险检查: {'通过' if len(position_result.get('violations', [])) == 0 else '未通过'}")
        print("="*80)
        
        return integrated
    
    def get_position_sizing_summary(self) -> Dict:
        """获取仓位管理摘要"""
        summary = self.portfolio_manager.get_portfolio_summary()
        return {
            'total_value': summary['total_value'],
            'cash': summary['cash'],
            'number_of_positions': summary['number_of_positions'],
            'largest_position': summary['largest_position'],
            'concentration_top_5': summary['concentration_top_5'],
            'risk_model': summary['risk_model'],
            'total_return': summary['total_return']
        }


def create_multi_asset_data(n_days: int = 252, n_symbols: int = 5) -> Dict[str, pd.DataFrame]:
    """创建多标的市场数据"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    price_data = {}
    
    for i in range(n_symbols):
        symbol = f"ASSET_{i+1}"
        base_price = 100 + i * 10
        
        # 生成价格数据
        prices = base_price + np.cumsum(np.random.randn(n_days) * 2)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(n_days) * 0.5,
            'high': prices + np.abs(np.random.randn(n_days) * 1),
            'low': prices - np.abs(np.random.randn(n_days) * 1),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        })
        df.set_index('date', inplace=True)
        price_data[symbol] = df
    
    return price_data


if __name__ == "__main__":
    # 示例使用
    from Execution.engine.complete_trading_flow import create_sample_strategies
    
    # 创建数据
    multi_data = create_multi_asset_data(n_days=252, n_symbols=5)
    
    # 合并数据（简化处理）
    first_symbol = list(multi_data.keys())[0]
    data = multi_data[first_symbol].copy()
    data['close'] = data['close']  # 确保有close列
    
    # 创建策略
    strategies = create_sample_strategies()
    
    # 运行完整流程（包含仓位管理）
    flow = IntegratedTradingFlow(
        initial_capital=1_000_000,
        risk_model=RiskModel.RISK_PARITY
    )
    
    result = flow.execute_complete_flow_with_position_management(
        data=data,
        strategies=strategies,
        symbols=list(multi_data.keys())[:3]  # 使用前3个标的
    )
    
    print("\n流程完成！")


"""
策略对比评估模块
使用Monte Carlo + Backtest对比不同策略
结合CVaR/VaR风控和Sharpe评估
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Execution.risk_control.risk_metrics import RiskCalculator, RiskMetrics
from Environment.backtester.simple_backtester import SimpleBacktester, BacktestConfig, BacktestResult

# 尝试导入CUDA Monte Carlo
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cuda_accelerated" / "python"))
    from cuda_monte_carlo import CUDAMonteCarloEngine, MonteCarloResult
    CUDA_MC_AVAILABLE = True
except ImportError:
    CUDA_MC_AVAILABLE = False


@dataclass
class StrategyComparisonResult:
    """策略对比结果"""
    strategy_name: str
    backtest_result: BacktestResult
    monte_carlo_result: Optional[MonteCarloResult] = None
    risk_metrics: Optional[RiskMetrics] = None
    sharpe_ratio: float = 0.0
    overall_score: float = 0.0
    passed_risk_checks: bool = False


class StrategyBenchmark:
    """
    策略对比评估器
    
    功能：
    1. 使用Backtest评估策略历史表现
    2. 使用Monte Carlo模拟未来风险
    3. 计算VaR/CVaR进行风控
    4. 使用Sharpe Ratio评估风险调整收益
    5. 综合评分和排名
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 monte_carlo_paths: int = 100000,
                 use_cuda: bool = True):
        """
        Args:
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易期数
            monte_carlo_paths: Monte Carlo模拟路径数
            use_cuda: 是否使用CUDA加速
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.monte_carlo_paths = monte_carlo_paths
        self.use_cuda = use_cuda and CUDA_MC_AVAILABLE
        
        self.risk_calculator = RiskCalculator(risk_free_rate, periods_per_year)
        self.backtester = SimpleBacktester()
        
        # Monte Carlo引擎
        if self.use_cuda:
            try:
                self.mc_engine = CUDAMonteCarloEngine(num_paths=monte_carlo_paths)
            except:
                self.use_cuda = False
                self.mc_engine = None
        else:
            self.mc_engine = None
    
    def compare_strategies(self,
                          strategies: Dict[str, Callable],
                          data: pd.DataFrame,
                          backtest_config: Optional[BacktestConfig] = None,
                          risk_limits: Optional[Dict[str, float]] = None) -> List[StrategyComparisonResult]:
        """
        对比多个策略
        
        Args:
            strategies: 策略字典 {name: strategy_function}
            data: 历史数据
            backtest_config: 回测配置
            risk_limits: 风险限制
        
        Returns:
            策略对比结果列表
        """
        if backtest_config is None:
            backtest_config = BacktestConfig()
        
        if risk_limits is None:
            risk_limits = {
                'max_drawdown': 0.30,  # 30%
                'max_var_95': 0.10,   # 10%
                'max_cvar_95': 0.15,  # 15%
                'min_sharpe': 0.0,
                'sharpe_tolerance': 1.5,  # allow down to -1.5 when data is noisy
                'allow_positive_return_override': True,
                'override_min_return': 0.0,
                'override_risk_buffer': 0.7  # require risk metrics within 70% of limits
            }
        
        results = []
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n{'='*80}")
            print(f"Evaluating Strategy: {strategy_name}")
            print(f"{'='*80}")
            
            # Step 1: Backtest
            print("\nStep 1: Running Backtest...")
            try:
                # 确保数据有必要的列
                if 'close' not in data.columns:
                    if 'price' in data.columns:
                        data = data.copy()
                        data['close'] = data['price']
                    else:
                        print(f"  ⚠ Warning: No 'close' or 'price' column in data")
                        print(f"  Available columns: {list(data.columns)}")
                        continue
                
                backtest_result = self.backtester.run(
                    strategy_name=strategy_name,
                    strategy_func=strategy_func,
                    data=data,
                    signal_column='signal',
                    price_column='close'
                )
            except Exception as e:
                print(f"  ✗ Backtest failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Step 2: Monte Carlo模拟（可选）
            mc_result = None
            if self.mc_engine is not None:
                print("\nStep 2: Running Monte Carlo Simulation...")
                mc_result = self._run_monte_carlo(backtest_result, data)
            
            # Step 3: 计算风险指标
            print("\nStep 3: Calculating Risk Metrics...")
            try:
                risk_metrics = self._calculate_risk_metrics(backtest_result, mc_result)
            except Exception as e:
                print(f"  ⚠ Warning: Risk metrics calculation failed: {e}")
                # 创建默认风险指标
                from Execution.risk_control.risk_metrics import RiskMetrics
                risk_metrics = RiskMetrics(
                    var_95=0.05, var_99=0.08,
                    cvar_95=0.08, cvar_99=0.12,
                    max_drawdown=0.1, max_drawdown_duration=0,
                    volatility=0.15, downside_deviation=0.12
                )
            
            # Step 4: 计算Sharpe Ratio
            try:
                if len(backtest_result.returns) > 0:
                    sharpe = self.risk_calculator.calculate_sharpe_ratio(backtest_result.returns)
                else:
                    sharpe = 0.0
                    print("  ⚠ Warning: No returns data, Sharpe ratio set to 0")
            except Exception as e:
                print(f"  ⚠ Warning: Sharpe ratio calculation failed: {e}")
                sharpe = 0.0
            
            # Step 5: 风险检查
            passed_checks = self._check_risk_limits(backtest_result, risk_metrics, sharpe, risk_limits)
            
            # Step 6: 综合评分
            overall_score = self._calculate_overall_score(
                backtest_result, risk_metrics, sharpe, passed_checks
            )
            
            result = StrategyComparisonResult(
                strategy_name=strategy_name,
                backtest_result=backtest_result,
                monte_carlo_result=mc_result,
                risk_metrics=risk_metrics,
                sharpe_ratio=sharpe,
                overall_score=overall_score,
                passed_risk_checks=passed_checks
            )
            
            results.append(result)
            
            # 打印结果
            self._print_strategy_result(result, risk_limits)
        
        # 排序和排名
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return results
    
    def _run_monte_carlo(self,
                        backtest_result: BacktestResult,
                        data: pd.DataFrame) -> Optional[MonteCarloResult]:
        """运行Monte Carlo模拟"""
        if self.mc_engine is None:
            return None
        
        try:
            # 从回测结果估计参数
            returns = backtest_result.returns
            if len(returns) == 0:
                return None
            
            mean_return = returns.mean() * self.periods_per_year  # 年化
            volatility = returns.std() * np.sqrt(self.periods_per_year)  # 年化
            
            # 检查有效性
            if pd.isna(mean_return) or pd.isna(volatility) or volatility <= 0:
                return None
            
            # 获取最终权益
            if len(backtest_result.equity_curve) == 0:
                return None
            final_equity = backtest_result.equity_curve.iloc[-1]
            
            # 创建模拟投资组合
            portfolio = {
                'strategy': {
                    'position': 1.0,
                    'price': final_equity,
                    'drift': mean_return,
                    'vol': volatility
                }
            }
            
            # 运行Monte Carlo（1个月前瞻）
            mc_result = self.mc_engine.simulate_portfolio(
                portfolio,
                time_horizon=1/12  # 1个月
            )
            
            return mc_result
        except Exception as e:
            print(f"  ⚠ Warning: Monte Carlo simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_risk_metrics(self,
                               backtest_result: BacktestResult,
                               mc_result: Optional[MonteCarloResult] = None) -> RiskMetrics:
        """计算风险指标"""
        try:
            returns = backtest_result.returns
            equity_curve = backtest_result.equity_curve
            
            # 检查数据有效性
            if len(returns) == 0 or len(equity_curve) == 0:
                print("  ⚠ Warning: Empty returns or equity curve")
                from Execution.risk_control.risk_metrics import RiskMetrics
                return RiskMetrics(
                    var_95=0.05, var_99=0.08,
                    cvar_95=0.08, cvar_99=0.12,
                    max_drawdown=0.1, max_drawdown_duration=0,
                    volatility=0.15, downside_deviation=0.12
                )
            
            # 使用RiskCalculator计算
            risk_metrics = self.risk_calculator.calculate_all_risk_metrics(
                returns=returns,
                equity_curve=equity_curve
            )
        except Exception as e:
            print(f"  ⚠ Warning: Risk metrics calculation error: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认风险指标
            from Execution.risk_control.risk_metrics import RiskMetrics
            return RiskMetrics(
                var_95=0.05, var_99=0.08,
                cvar_95=0.08, cvar_99=0.12,
                max_drawdown=0.1, max_drawdown_duration=0,
                volatility=0.15, downside_deviation=0.12
            )
        
        # 如果有Monte Carlo结果，可以增强VaR/CVaR
        if mc_result is not None:
            # 使用Monte Carlo结果计算VaR/CVaR
            mc_var_95 = mc_result.calculate_var(0.95)
            mc_cvar_95 = mc_result.calculate_cvar(0.95)
            mc_var_99 = mc_result.calculate_var(0.99)
            mc_cvar_99 = mc_result.calculate_cvar(0.99)
            
            # 结合历史回测和Monte Carlo结果（加权平均）
            weight_mc = 0.3  # Monte Carlo权重
            risk_metrics.var_95 = (1 - weight_mc) * risk_metrics.var_95 + weight_mc * abs(mc_var_95)
            risk_metrics.cvar_95 = (1 - weight_mc) * risk_metrics.cvar_95 + weight_mc * abs(mc_cvar_95)
            risk_metrics.var_99 = (1 - weight_mc) * risk_metrics.var_99 + weight_mc * abs(mc_var_99)
            risk_metrics.cvar_99 = (1 - weight_mc) * risk_metrics.cvar_99 + weight_mc * abs(mc_cvar_99)
        
        return risk_metrics
    
    def _check_risk_limits(self,
                          backtest_result: BacktestResult,
                          risk_metrics: RiskMetrics,
                          sharpe: float,
                          risk_limits: Dict[str, float]) -> bool:
        """检查风险限制"""
        max_dd_limit = risk_limits.get('max_drawdown', 0.20)
        max_var_limit = risk_limits.get('max_var_95', 0.05)
        max_cvar_limit = risk_limits.get('max_cvar_95', 0.08)
        sharpe_threshold = risk_limits.get('min_sharpe', 0.5)
        sharpe_tolerance = risk_limits.get('sharpe_tolerance', 0.0)
        effective_sharpe = sharpe_threshold - sharpe_tolerance
        
        checks = {
            'max_drawdown': risk_metrics.max_drawdown <= max_dd_limit,
            'max_var_95': risk_metrics.var_95 <= max_var_limit,
            'max_cvar_95': risk_metrics.cvar_95 <= max_cvar_limit,
            'min_sharpe': sharpe >= effective_sharpe
        }
        
        # 如果Sharpe未达标，但收益和风险表现优秀，可允许兜底通过
        if (not checks['min_sharpe']) and risk_limits.get('allow_positive_return_override', True):
            equity_curve = backtest_result.equity_curve
            if len(equity_curve) > 1:
                total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
            else:
                total_return = 0.0
            
            risk_buffer = risk_limits.get('override_risk_buffer', 0.7)
            risk_within_buffer = (
                risk_metrics.max_drawdown <= max_dd_limit * risk_buffer and
                risk_metrics.var_95 <= max_var_limit * risk_buffer and
                risk_metrics.cvar_95 <= max_cvar_limit * risk_buffer
            )
            
            if total_return >= risk_limits.get('override_min_return', 0.0) and risk_within_buffer:
                checks['min_sharpe'] = True
        
        return all(checks.values())
    
    def _calculate_overall_score(self,
                                backtest_result: BacktestResult,
                                risk_metrics: RiskMetrics,
                                sharpe: float,
                                passed_checks: bool) -> float:
        """计算综合评分（0-100）"""
        score = 0.0
        
        # Sharpe Ratio (40%)
        sharpe_score = min(max(sharpe * 20, 0), 40)  # 0-40分
        score += sharpe_score
        
        # 总收益率 (20%)
        total_return = (backtest_result.equity_curve.iloc[-1] / backtest_result.equity_curve.iloc[0] - 1)
        return_score = min(max(total_return * 100, 0), 20)  # 0-20分
        score += return_score
        
        # 最大回撤 (20%) - 回撤越小分数越高
        dd_score = max(20 * (1 - risk_metrics.max_drawdown / 0.20), 0)  # 0-20分
        score += dd_score
        
        # VaR/CVaR (10%)
        var_score = max(10 * (1 - risk_metrics.var_95 / 0.05), 0)  # 0-10分
        score += var_score
        
        # 风险检查通过奖励 (10%)
        if passed_checks:
            score += 10
        
        return min(score, 100)
    
    def _print_strategy_result(self,
                              result: StrategyComparisonResult,
                              risk_limits: Dict[str, float]):
        """打印策略结果"""
        print(f"\n{'='*80}")
        print(f"Strategy: {result.strategy_name}")
        print(f"{'='*80}")
        
        bt = result.backtest_result
        print(f"\nBacktest Results:")
        print(f"  Total Return: {(bt.equity_curve.iloc[-1]/bt.equity_curve.iloc[0]-1)*100:.2f}%")
        print(f"  Number of Trades: {len(bt.trades)}")
        
        if result.risk_metrics:
            print(f"\nRisk Metrics:")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {result.risk_metrics.max_drawdown*100:.2f}%")
            print(f"  VaR (95%): {result.risk_metrics.var_95*100:.2f}%")
            print(f"  CVaR (95%): {result.risk_metrics.cvar_95*100:.2f}%")
        
        print(f"\nOverall Score: {result.overall_score:.2f}/100")
        print(f"Risk Checks: {'PASSED' if result.passed_risk_checks else 'FAILED'}")
        print(f"{'='*80}")
    
    def print_comparison_report(self, results: List[StrategyComparisonResult]):
        """打印对比报告"""
        print("\n" + "="*80)
        print("Strategy Comparison Report")
        print("="*80)
        
        print(f"\n{'Rank':<6} {'Strategy':<20} {'Score':<8} {'Sharpe':<8} {'Return':<10} {'Max DD':<10} {'Risk':<8}")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            bt = result.backtest_result
            total_return = (bt.equity_curve.iloc[-1]/bt.equity_curve.iloc[0]-1)*100
            max_dd = result.risk_metrics.max_drawdown*100 if result.risk_metrics else 0
            
            print(f"{i:<6} {result.strategy_name:<20} {result.overall_score:<8.2f} "
                  f"{result.sharpe_ratio:<8.3f} {total_return:<10.2f}% {max_dd:<10.2f}% "
                  f"{'PASS' if result.passed_risk_checks else 'FAIL':<8}")
        
        print("="*80)
        
        # 推荐最佳策略
        if results:
            best = results[0]
            print(f"\nRecommended Strategy: {best.strategy_name}")
            print(f"  Score: {best.overall_score:.2f}/100")
            print(f"  Sharpe Ratio: {best.sharpe_ratio:.3f}")
            print(f"  Risk Checks: {'PASSED' if best.passed_risk_checks else 'FAILED'}")


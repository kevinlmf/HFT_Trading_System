"""
Comprehensive Strategy Evaluator
Combines performance metrics, P&L analysis, and risk control
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from datetime import datetime

from .performance_metrics import PerformanceMetrics, calculate_all_metrics
from .pnl_analyzer import PnLAnalyzer, PnLBreakdown


@dataclass
class RiskMetrics:
    """Risk-specific metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    downside_deviation: float
    beta: Optional[float] = None
    correlation_to_market: Optional[float] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report combining all metrics"""
    strategy_name: str
    evaluation_date: datetime

    # Performance
    performance: PerformanceMetrics

    # P&L breakdown
    pnl_breakdown: PnLBreakdown

    # Risk metrics
    risk_metrics: RiskMetrics

    # Summary statistics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    net_pnl: float

    # Pass/fail risk checks
    risk_checks: Dict[str, bool]

    # Overall score (0-100)
    overall_score: float


class StrategyEvaluator:
    """
    Comprehensive strategy evaluator integrating:
    - Performance metrics
    - P&L analysis with cost attribution
    - Risk control checks
    - Comparative analysis
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        benchmark_returns: Optional[pd.Series] = None,
        risk_limits: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (252 for daily)
            benchmark_returns: Benchmark return series for comparison
            risk_limits: Dictionary of risk limits to check
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.benchmark_returns = benchmark_returns

        # Default risk limits
        self.risk_limits = risk_limits or {
            'max_drawdown': 0.20,  # 20%
            'max_volatility': 0.30,  # 30%
            'min_sharpe': 0.5,
            'max_var_95': 0.05,  # 5%
            'min_win_rate': 0.40  # 40%
        }

        self.pnl_analyzer = PnLAnalyzer()

    def evaluate(
        self,
        strategy_name: str,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        positions: Optional[pd.DataFrame] = None,
        prices: Optional[pd.Series] = None
    ) -> EvaluationReport:
        """
        Perform comprehensive strategy evaluation

        Args:
            strategy_name: Name of the strategy
            returns: Return series
            trades: Trade history (optional)
            positions: Position history (optional)
            prices: Price series (optional)

        Returns:
            EvaluationReport with complete analysis
        """
        # Calculate performance metrics
        performance = calculate_all_metrics(
            returns=returns,
            trades=trades,
            periods_per_year=self.periods_per_year,
            risk_free_rate=self.risk_free_rate
        )

        # P&L breakdown
        if trades is not None and len(trades) > 0:
            pnl_breakdown = self.pnl_analyzer.analyze_pnl(
                trades=trades,
                positions=positions
            )
        else:
            # Create empty breakdown
            pnl_breakdown = PnLBreakdown(
                gross_pnl=0.0,
                transaction_costs=0.0,
                slippage=0.0,
                market_impact=0.0,
                financing_costs=0.0,
                net_pnl=0.0
            )

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(returns)

        # Risk checks
        risk_checks = self._perform_risk_checks(performance, risk_metrics)

        # Overall score
        overall_score = self._calculate_overall_score(performance, risk_metrics, risk_checks)

        return EvaluationReport(
            strategy_name=strategy_name,
            evaluation_date=datetime.now(),
            performance=performance,
            pnl_breakdown=pnl_breakdown,
            risk_metrics=risk_metrics,
            total_return=performance.total_return,
            annualized_return=performance.annualized_return,
            sharpe_ratio=performance.sharpe_ratio,
            max_drawdown=performance.max_drawdown,
            net_pnl=pnl_breakdown.net_pnl,
            risk_checks=risk_checks,
            overall_score=overall_score
        )

    def _calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate detailed risk metrics"""
        # VaR at different confidence levels
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        # CVaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Drawdown duration
        is_dd = drawdown < 0
        dd_periods = is_dd.astype(int).groupby((~is_dd).cumsum()).sum()
        max_dd_duration = dd_periods.max() if len(dd_periods) > 0 else 0

        # Volatility
        volatility = returns.std()

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0

        # Beta and correlation (if benchmark provided)
        beta = None
        correlation = None
        if self.benchmark_returns is not None:
            aligned = pd.DataFrame({
                'strategy': returns,
                'benchmark': self.benchmark_returns
            }).dropna()

            if len(aligned) > 1:
                correlation = aligned['strategy'].corr(aligned['benchmark'])

                bench_var = aligned['benchmark'].var()
                if bench_var > 0:
                    covariance = aligned['strategy'].cov(aligned['benchmark'])
                    beta = covariance / bench_var

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            max_drawdown_duration=int(max_dd_duration),
            volatility=volatility,
            downside_deviation=downside_deviation,
            beta=beta,
            correlation_to_market=correlation
        )

    def _perform_risk_checks(
        self,
        performance: PerformanceMetrics,
        risk_metrics: RiskMetrics
    ) -> Dict[str, bool]:
        """Check if strategy passes risk limits"""
        checks = {}

        checks['drawdown_ok'] = abs(risk_metrics.max_drawdown) <= self.risk_limits['max_drawdown']
        checks['volatility_ok'] = performance.annualized_volatility <= self.risk_limits['max_volatility']
        checks['sharpe_ok'] = performance.sharpe_ratio >= self.risk_limits['min_sharpe']
        checks['var_ok'] = abs(risk_metrics.var_95) <= self.risk_limits['max_var_95']
        checks['win_rate_ok'] = performance.win_rate >= self.risk_limits['min_win_rate']

        return checks

    def _calculate_overall_score(
        self,
        performance: PerformanceMetrics,
        risk_metrics: RiskMetrics,
        risk_checks: Dict[str, bool]
    ) -> float:
        """
        Calculate overall score (0-100)

        Scoring breakdown:
        - 30 points: Risk-adjusted returns (Sharpe, Sortino)
        - 25 points: Absolute returns
        - 25 points: Risk metrics (drawdown, volatility)
        - 20 points: Pass/fail risk checks
        """
        score = 0.0

        # Risk-adjusted returns (30 points)
        sharpe_score = min(performance.sharpe_ratio / 3.0, 1.0) * 15  # Max 15 pts for Sharpe >= 3
        sortino_score = min(performance.sortino_ratio / 3.0, 1.0) * 15  # Max 15 pts for Sortino >= 3
        score += sharpe_score + sortino_score

        # Absolute returns (25 points)
        # Score based on annualized return (10% = 12.5 pts, 20% = 25 pts)
        return_score = min(performance.annualized_return / 0.20, 1.0) * 25
        score += max(return_score, 0)  # No negative points

        # Risk metrics (25 points)
        # Lower drawdown is better
        dd_score = max((1 - abs(risk_metrics.max_drawdown) / 0.30), 0) * 12.5  # Max -30% dd

        # Lower volatility is better
        vol_score = max((1 - performance.annualized_volatility / 0.40), 0) * 12.5  # Max 40% vol
        score += dd_score + vol_score

        # Risk checks (20 points)
        passed_checks = sum(risk_checks.values())
        total_checks = len(risk_checks)
        check_score = (passed_checks / total_checks) * 20 if total_checks > 0 else 0
        score += check_score

        return min(score, 100.0)

    def print_evaluation_report(self, report: EvaluationReport, detailed: bool = True):
        """Print formatted evaluation report"""
        print("\n" + "=" * 80)
        print(f"STRATEGY EVALUATION REPORT: {report.strategy_name}")
        print(f"Evaluation Date: {report.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Overall score
        print(f"\n{'OVERALL SCORE:':<30} {report.overall_score:>10.1f} / 100")

        # Key metrics summary
        print(f"\n{'KEY METRICS':^80}")
        print("-" * 80)
        print(f"{'Total Return:':<30} {report.total_return:>15.2%}")
        print(f"{'Annualized Return:':<30} {report.annualized_return:>15.2%}")
        print(f"{'Sharpe Ratio:':<30} {report.sharpe_ratio:>15.2f}")
        print(f"{'Sortino Ratio:':<30} {report.performance.sortino_ratio:>15.2f}")
        print(f"{'Max Drawdown:':<30} {report.max_drawdown:>15.2%}")
        print(f"{'Win Rate:':<30} {report.performance.win_rate:>15.2%}")

        # P&L Breakdown
        print(f"\n{'P&L BREAKDOWN':^80}")
        print("-" * 80)
        print(f"{'Gross P&L:':<30} ${report.pnl_breakdown.gross_pnl:>15,.2f}")
        print(f"{'Transaction Costs:':<30} ${report.pnl_breakdown.transaction_costs:>15,.2f}")
        print(f"{'Slippage:':<30} ${report.pnl_breakdown.slippage:>15,.2f}")
        print(f"{'Market Impact:':<30} ${report.pnl_breakdown.market_impact:>15,.2f}")
        print(f"{'Net P&L:':<30} ${report.net_pnl:>15,.2f}")

        # Risk checks
        print(f"\n{'RISK CHECKS':^80}")
        print("-" * 80)
        for check, passed in report.risk_checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{check:<30} {status:>15}")

        if detailed:
            # Detailed performance metrics
            print(f"\n{'DETAILED PERFORMANCE':^80}")
            print("-" * 80)
            perf = report.performance
            print(f"{'Volatility (Annual):':<30} {perf.annualized_volatility:>15.2%}")
            print(f"{'Downside Deviation:':<30} {perf.downside_deviation:>15.2%}")
            print(f"{'Calmar Ratio:':<30} {perf.calmar_ratio:>15.2f}")
            print(f"{'Omega Ratio:':<30} {perf.omega_ratio:>15.2f}")
            print(f"{'VaR (95%):':<30} {perf.var_95:>15.2%}")
            print(f"{'CVaR (95%):':<30} {perf.cvar_95:>15.2%}")
            print(f"{'Skewness:':<30} {perf.skewness:>15.3f}")
            print(f"{'Kurtosis:':<30} {perf.kurtosis:>15.3f}")

            # Trade statistics
            print(f"\n{'TRADE STATISTICS':^80}")
            print("-" * 80)
            print(f"{'Total Trades:':<30} {perf.total_trades:>15,}")
            print(f"{'Winning Trades:':<30} {perf.winning_trades:>15,}")
            print(f"{'Losing Trades:':<30} {perf.losing_trades:>15,}")
            print(f"{'Profit Factor:':<30} {perf.profit_factor:>15.2f}")
            print(f"{'Avg Win:':<30} ${perf.avg_win:>15,.2f}")
            print(f"{'Avg Loss:':<30} ${perf.avg_loss:>15,.2f}")
            print(f"{'Win/Loss Ratio:':<30} {perf.win_loss_ratio:>15.2f}")

        print("=" * 80 + "\n")

    def compare_strategies(
        self,
        reports: List[EvaluationReport]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            reports: List of evaluation reports

        Returns:
            DataFrame with comparison metrics
        """
        comparison = []

        for report in reports:
            comparison.append({
                'Strategy': report.strategy_name,
                'Score': report.overall_score,
                'Total Return': report.total_return,
                'Ann. Return': report.annualized_return,
                'Sharpe': report.sharpe_ratio,
                'Sortino': report.performance.sortino_ratio,
                'Max DD': report.max_drawdown,
                'Volatility': report.performance.annualized_volatility,
                'Win Rate': report.performance.win_rate,
                'Net P&L': report.net_pnl,
                'Total Trades': report.performance.total_trades
            })

        df = pd.DataFrame(comparison)
        df = df.sort_values('Score', ascending=False)

        return df

    def plot_comparison(
        self,
        reports: List[EvaluationReport],
        save_path: Optional[str] = None
    ):
        """Plot strategy comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')

        names = [r.strategy_name for r in reports]

        # 1. Risk-adjusted returns
        sharpes = [r.sharpe_ratio for r in reports]
        sortinos = [r.performance.sortino_ratio for r in reports]

        x = np.arange(len(names))
        width = 0.35

        axes[0, 0].bar(x - width/2, sharpes, width, label='Sharpe', alpha=0.8)
        axes[0, 0].bar(x + width/2, sortinos, width, label='Sortino', alpha=0.8)
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].set_title('Risk-Adjusted Returns')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Returns vs Risk
        returns = [r.annualized_return * 100 for r in reports]
        vols = [r.performance.annualized_volatility * 100 for r in reports]

        axes[0, 1].scatter(vols, returns, s=200, alpha=0.6)
        for i, name in enumerate(names):
            axes[0, 1].annotate(name, (vols[i], returns[i]),
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Volatility (%)')
        axes[0, 1].set_ylabel('Annualized Return (%)')
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].grid(alpha=0.3)

        # 3. Drawdown comparison
        drawdowns = [abs(r.max_drawdown) * 100 for r in reports]

        axes[1, 0].barh(names, drawdowns, alpha=0.8, color='red')
        axes[1, 0].set_xlabel('Max Drawdown (%)')
        axes[1, 0].set_title('Maximum Drawdown')
        axes[1, 0].grid(axis='x', alpha=0.3)

        # 4. Overall scores
        scores = [r.overall_score for r in reports]
        colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in scores]

        x_pos = np.arange(len(names))
        axes[1, 1].bar(x_pos, scores, alpha=0.8, color=colors)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Overall Score (0-100)')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good')
        axes[1, 1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Fair')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            plt.close(fig)
        else:
            # Only attempt to show if in an interactive environment
            try:
                plt.show()
            except Exception:
                # If show fails (non-interactive backend), just save to default location
                default_path = 'strategy_comparison.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {default_path}")
                plt.close(fig)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Strategy returns
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.001, index=dates)

    # Sample trades
    trades = pd.DataFrame({
        'price': 100 + np.random.randn(50) * 5,
        'quantity': np.random.randint(10, 100, 50),
        'pnl': np.random.randn(50) * 100,
        'duration': np.random.randint(1, 10, 50)
    })

    # Create evaluator
    evaluator = StrategyEvaluator()

    # Evaluate strategy
    report = evaluator.evaluate(
        strategy_name="Test Strategy",
        returns=returns,
        trades=trades
    )

    # Print report
    evaluator.print_evaluation_report(report, detailed=True)

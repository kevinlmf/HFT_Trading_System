"""
完整的风险指标计算模块
包括VaR、CVaR、Sharpe Ratio等
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """完整的风险指标"""
    # VaR (Value at Risk)
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    
    # CVaR (Conditional VaR / Expected Shortfall)
    cvar_95: float  # 95% CVaR
    cvar_99: float  # 99% CVaR
    
    # 其他风险指标
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    downside_deviation: float
    
    # 相关性指标
    beta: Optional[float] = None
    correlation_to_market: Optional[float] = None
    
    # Sharpe相关
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None


class RiskCalculator:
    """
    风险指标计算器
    
    功能：
    - VaR计算（历史模拟法、参数法）
    - CVaR计算
    - Sharpe Ratio计算
    - 最大回撤计算
    - 其他风险指标
    """
    
    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年交易期数（252为日度）
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def calculate_var(self,
                     returns: pd.Series,
                     confidence: float = 0.95,
                     method: str = "historical") -> float:
        """
        计算VaR (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence: 置信水平 (0.95 或 0.99)
            method: 计算方法 ("historical" 或 "parametric")
        
        Returns:
            VaR值（正数表示损失）
        """
        if method == "historical":
            # 历史模拟法
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            # 参数法（假设正态分布）
            mean = returns.mean()
            std = returns.std()
            try:
                from scipy import stats
                z_score = stats.norm.ppf(1 - confidence)
            except ImportError:
                # Fallback: 使用近似值
                z_score = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
            return -(mean + z_score * std)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_cvar(self,
                      returns: pd.Series,
                      confidence: float = 0.95,
                      method: str = "historical") -> float:
        """
        计算CVaR (Conditional VaR / Expected Shortfall)
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            method: 计算方法
        
        Returns:
            CVaR值（正数表示损失）
        """
        if method == "historical":
            # 历史模拟法
            var = self.calculate_var(returns, confidence, method="historical")
            # CVaR是超过VaR的损失的期望值
            tail_losses = returns[returns <= -var]
            if len(tail_losses) > 0:
                return -tail_losses.mean()
            else:
                return var
        elif method == "parametric":
            # 参数法
            mean = returns.mean()
            std = returns.std()
            try:
                from scipy import stats
                z_score = stats.norm.ppf(1 - confidence)
                # CVaR公式（正态分布）
                cvar = -(mean + std * stats.norm.pdf(z_score) / (1 - confidence))
            except ImportError:
                # Fallback: 使用近似值
                z_score = {0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
                cvar = -(mean + std * 0.103 / (1 - confidence))  # 近似值
            return cvar
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_sharpe_ratio(self,
                               returns: pd.Series,
                               annualized: bool = True) -> float:
        """
        计算Sharpe Ratio
        
        Args:
            returns: 收益率序列
            annualized: 是否年化
        
        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        sharpe = excess_returns.mean() / returns.std()
        
        if annualized:
            sharpe *= np.sqrt(self.periods_per_year)
        
        return sharpe
    
    def calculate_sortino_ratio(self,
                               returns: pd.Series,
                               annualized: bool = True) -> float:
        """
        计算Sortino Ratio（只考虑下行风险）
        
        Args:
            returns: 收益率序列
            annualized: 是否年化
        
        Returns:
            Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / self.periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std
        
        if annualized:
            sortino *= np.sqrt(self.periods_per_year)
        
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线（资产价值序列）
        
        Returns:
            (max_drawdown, max_drawdown_duration)
        """
        if len(equity_curve) == 0:
            return 0.0, 0
        
        # 计算累积最大值
        cummax = equity_curve.expanding().max()
        
        # 计算回撤
        drawdown = (equity_curve - cummax) / cummax
        
        max_dd = drawdown.min()
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        return abs(max_dd), max_dd_duration
    
    def calculate_calmar_ratio(self,
                              returns: pd.Series,
                              equity_curve: pd.Series,
                              annualized: bool = True) -> float:
        """
        计算Calmar Ratio (年化收益率 / 最大回撤)
        
        Args:
            returns: 收益率序列
            equity_curve: 权益曲线
            annualized: 是否年化收益率
        
        Returns:
            Calmar Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * self.periods_per_year if annualized else returns.mean()
        max_dd, _ = self.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    def calculate_all_risk_metrics(self,
                                   returns: pd.Series,
                                   equity_curve: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        计算所有风险指标
        
        Args:
            returns: 收益率序列
            equity_curve: 权益曲线
            benchmark_returns: 基准收益率（可选，用于计算beta和相关性）
        
        Returns:
            RiskMetrics对象
        """
        # VaR
        var_95 = self.calculate_var(returns, confidence=0.95)
        var_99 = self.calculate_var(returns, confidence=0.99)
        
        # CVaR
        cvar_95 = self.calculate_cvar(returns, confidence=0.95)
        cvar_99 = self.calculate_cvar(returns, confidence=0.99)
        
        # 最大回撤
        max_dd, max_dd_duration = self.calculate_max_drawdown(equity_curve)
        
        # 波动率
        volatility = returns.std() * np.sqrt(self.periods_per_year)
        
        # 下行标准差
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.periods_per_year) if len(downside_returns) > 0 else 0.0
        
        # Sharpe Ratio
        sharpe = self.calculate_sharpe_ratio(returns)
        
        # Sortino Ratio
        sortino = self.calculate_sortino_ratio(returns)
        
        # Calmar Ratio
        calmar = self.calculate_calmar_ratio(returns, equity_curve)
        
        # Beta和相关性（如果有基准）
        beta = None
        correlation = None
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            correlation = returns.corr(benchmark_returns)
            if benchmark_returns.std() > 0:
                beta = returns.cov(benchmark_returns) / benchmark_returns.var()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            downside_deviation=downside_deviation,
            beta=beta,
            correlation_to_market=correlation,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar
        )
    
    def print_risk_report(self, metrics: RiskMetrics):
        """打印风险报告"""
        print("=" * 80)
        print("Risk Metrics Report")
        print("=" * 80)
        print(f"\nVaR (Value at Risk):")
        print(f"  VaR (95%): {metrics.var_95*100:.2f}%")
        print(f"  VaR (99%): {metrics.var_99*100:.2f}%")
        
        print(f"\nCVaR (Conditional VaR):")
        print(f"  CVaR (95%): {metrics.cvar_95*100:.2f}%")
        print(f"  CVaR (99%): {metrics.cvar_99*100:.2f}%")
        
        print(f"\nDrawdown:")
        print(f"  Max Drawdown: {metrics.max_drawdown*100:.2f}%")
        print(f"  Max Drawdown Duration: {metrics.max_drawdown_duration} periods")
        
        print(f"\nVolatility:")
        print(f"  Annual Volatility: {metrics.volatility*100:.2f}%")
        print(f"  Downside Deviation: {metrics.downside_deviation*100:.2f}%")
        
        print(f"\nRisk-Adjusted Returns:")
        if metrics.sharpe_ratio is not None:
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        if metrics.sortino_ratio is not None:
            print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
        if metrics.calmar_ratio is not None:
            print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
        
        if metrics.beta is not None:
            print(f"\nMarket Correlation:")
            print(f"  Beta: {metrics.beta:.3f}")
            print(f"  Correlation: {metrics.correlation_to_market:.3f}")
        
        print("=" * 80)


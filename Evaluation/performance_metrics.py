"""
Comprehensive Performance Metrics for Trading Strategies
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Container for strategy performance metrics"""

    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk
    volatility: float
    annualized_volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float

    # Advanced metrics
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    skewness: float
    kurtosis: float

    # Timing
    best_day: float
    worst_day: float
    best_month: float
    worst_month: float

    # Consistency
    monthly_win_rate: float
    consecutive_wins: int
    consecutive_losses: int


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from price series"""
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns"""
    return (1 + returns).cumprod() - 1


def calculate_drawdown(cumulative_returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdown series, maximum drawdown, and duration

    Returns:
        (drawdown_series, max_drawdown, max_drawdown_duration)
    """
    wealth_index = 1 + cumulative_returns
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    max_dd = drawdowns.min()

    # Calculate max drawdown duration
    is_in_drawdown = drawdowns < 0
    drawdown_periods = is_in_drawdown.astype(int).groupby(
        (~is_in_drawdown).cumsum()
    ).sum()
    max_dd_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0

    return drawdowns, max_dd, int(max_dd_duration)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio (using downside deviation)"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    """Calculate Calmar ratio"""
    if max_drawdown == 0:
        return 0.0
    return annualized_return / abs(max_drawdown)


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio
    Ratio of probability-weighted gains to probability-weighted losses
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()

    if losses == 0:
        return float('inf') if gains > 0 else 0.0

    return gains / losses


def calculate_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk and Conditional VaR

    Returns:
        (VaR, CVaR) at given confidence level
    """
    var = returns.quantile(1 - confidence)
    cvar = returns[returns <= var].mean()
    return var, cvar


def calculate_trade_metrics(
    trades: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate trading metrics from trade history

    Args:
        trades: DataFrame with columns ['entry_price', 'exit_price', 'quantity', 'pnl', 'duration']

    Returns:
        Dictionary of trade metrics
    """
    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_duration': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }

    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]

    total_trades = len(trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)

    win_rate = num_winning / total_trades if total_trades > 0 else 0.0

    total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
    total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0

    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0

    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    avg_duration = trades['duration'].mean() if 'duration' in trades.columns else 0.0

    # Calculate consecutive wins/losses
    is_win = (trades['pnl'] > 0).astype(int)
    consecutive_wins = (is_win * (is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1)).max()

    is_loss = (trades['pnl'] < 0).astype(int)
    consecutive_losses = (is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)).max()

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'total_trades': total_trades,
        'winning_trades': num_winning,
        'losing_trades': num_losing,
        'avg_trade_duration': avg_duration,
        'consecutive_wins': int(consecutive_wins),
        'consecutive_losses': int(consecutive_losses)
    }


def calculate_monthly_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate monthly aggregated metrics"""
    if not isinstance(returns.index, pd.DatetimeIndex):
        return {
            'best_month': 0.0,
            'worst_month': 0.0,
            'monthly_win_rate': 0.0
        }

    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0.0
    worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0.0

    winning_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    monthly_win_rate = winning_months / total_months if total_months > 0 else 0.0

    return {
        'best_month': best_month,
        'worst_month': worst_month,
        'monthly_win_rate': monthly_win_rate
    }


def calculate_all_metrics(
    prices: Optional[pd.Series] = None,
    returns: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate all performance metrics

    Args:
        prices: Price series (optional if returns provided)
        returns: Return series (optional if prices provided)
        trades: Trade history DataFrame (optional)
        periods_per_year: Number of trading periods per year (252 for daily)
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object with all metrics
    """
    # Calculate returns if not provided
    if returns is None and prices is not None:
        returns = calculate_returns(prices)
    elif returns is None:
        raise ValueError("Either prices or returns must be provided")

    if len(returns) == 0:
        raise ValueError("Returns series is empty")

    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    cumulative_returns = calculate_cumulative_returns(returns)
    cumulative_return = cumulative_returns.iloc[-1]

    # Risk metrics
    volatility = returns.std()
    annualized_volatility = volatility * np.sqrt(periods_per_year)

    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0

    drawdowns, max_drawdown, max_dd_duration = calculate_drawdown(cumulative_returns)

    # Risk-adjusted returns
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(annualized_return, max_drawdown)
    omega = calculate_omega_ratio(returns)

    # VaR and CVaR
    var_95, cvar_95 = calculate_var_cvar(returns, 0.95)

    # Distribution metrics
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Best/worst days
    best_day = returns.max()
    worst_day = returns.min()

    # Monthly metrics
    monthly_metrics = calculate_monthly_metrics(returns)

    # Trade metrics
    if trades is not None and len(trades) > 0:
        trade_metrics = calculate_trade_metrics(trades)
    else:
        trade_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_duration': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        cumulative_return=cumulative_return,
        volatility=volatility,
        annualized_volatility=annualized_volatility,
        downside_deviation=downside_deviation,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        var_95=var_95,
        cvar_95=cvar_95,
        skewness=skewness,
        kurtosis=kurtosis,
        best_day=best_day,
        worst_day=worst_day,
        best_month=monthly_metrics['best_month'],
        worst_month=monthly_metrics['worst_month'],
        monthly_win_rate=monthly_metrics['monthly_win_rate'],
        **trade_metrics
    )


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005, index=dates)

    # Calculate metrics
    metrics = calculate_all_metrics(returns=returns)

    print("Performance Metrics:")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Annualized Volatility: {metrics.annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"VaR (95%): {metrics.var_95:.2%}")
    print(f"CVaR (95%): {metrics.cvar_95:.2%}")

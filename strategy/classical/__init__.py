"""
Classical Trading Strategies

This module contains traditional quantitative trading strategies:
- Momentum: Trend-following strategies
- Mean Variance: Portfolio optimization
- Pairs Trading: Statistical arbitrage on cointegrated pairs
- Statistical Arbitrage: Market-neutral strategies
"""

from .momentum_strategy import MomentumStrategy
from .mean_variance import MeanVarianceStrategy
from .pairs_trading import PairsTradingStrategy
from .statistical_arbitrage import StatisticalArbitrageStrategy

__all__ = [
    'MomentumStrategy',
    'MeanVarianceStrategy',
    'PairsTradingStrategy',
    'StatisticalArbitrageStrategy',
]

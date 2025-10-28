"""
Strategy Adapter
Provides unified interface for different strategy implementations
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class StrategyInfo:
    """Information about a strategy"""
    name: str
    category: str
    description: str
    signal_generator: Callable[[pd.DataFrame], float]


class StrategyAdapter:
    """
    Adapter to create unified signal generators from various strategy implementations

    Handles strategies with different interfaces:
    - Momentum-based strategies
    - Mean reversion strategies
    - Statistical arbitrage
    - Market making
    - ML-based strategies
    """

    @staticmethod
    def adapt_momentum_strategy(
        lookback: int = 20,
        threshold: float = 0.001
    ) -> StrategyInfo:
        """Create momentum strategy"""
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < lookback:
                return 0
            returns = data['close'].pct_change()
            momentum = returns.tail(lookback).mean()
            return 1 if momentum > threshold else (-1 if momentum < -threshold else 0)

        return StrategyInfo(
            name='Momentum',
            category='Trend',
            description=f'Momentum strategy with {lookback}-period lookback',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_mean_reversion_strategy(
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> StrategyInfo:
        """Create mean reversion strategy"""
        position = [0]  # Mutable to track state

        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < lookback:
                return 0
            prices = data['close']
            mean = prices.tail(lookback).mean()
            std = prices.tail(lookback).std()
            current = prices.iloc[-1]

            if std == 0:
                return 0

            z_score = (current - mean) / std

            # Entry signals
            if z_score > entry_threshold:
                position[0] = -1  # Short
                return -1
            elif z_score < -entry_threshold:
                position[0] = 1  # Long
                return 1

            # Exit signals
            if abs(z_score) < exit_threshold:
                if position[0] != 0:
                    position[0] = 0
                    return 0

            return position[0]

        return StrategyInfo(
            name='Mean Reversion',
            category='Mean Reversion',
            description=f'Mean reversion with {lookback}-period lookback, entry Z={entry_threshold}',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_statistical_arbitrage(
        short_window: int = 20,
        long_window: int = 60
    ) -> StrategyInfo:
        """Create statistical arbitrage strategy"""
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < long_window:
                return 0
            prices = data['close']
            sma_short = prices.tail(short_window).mean()
            sma_long = prices.tail(long_window).mean()
            return 1 if sma_short > sma_long * 1.01 else (-1 if sma_short < sma_long * 0.99 else 0)

        return StrategyInfo(
            name='Statistical Arbitrage',
            category='Arbitrage',
            description=f'Stat arb with SMA({short_window}/{long_window})',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_pairs_trading(
        lookback: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5
    ) -> StrategyInfo:
        """Create pairs trading strategy (simplified single-asset version)"""
        position = [0]

        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < lookback:
                return 0

            # Use price deviation from its own mean as proxy
            prices = data['close']
            mean = prices.tail(lookback).mean()
            std = prices.tail(lookback).std()
            current = prices.iloc[-1]

            if std == 0:
                return 0

            z_score = (current - mean) / std

            # Entry
            if abs(z_score) > z_entry:
                position[0] = -np.sign(z_score)
                return position[0]

            # Exit
            if abs(z_score) < z_exit and position[0] != 0:
                position[0] = 0
                return 0

            return position[0]

        return StrategyInfo(
            name='Pairs Trading',
            category='Arbitrage',
            description=f'Pairs trading with {lookback}-period lookback',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_market_making(
        rsi_period: int = 14,
        overbought: float = 70,
        oversold: float = 30
    ) -> StrategyInfo:
        """Create market making strategy (using RSI as proxy)"""
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < rsi_period + 1:
                return 0

            # Calculate RSI
            prices = data['close']
            deltas = prices.diff()
            gain = (deltas.where(deltas > 0, 0)).tail(rsi_period).mean()
            loss = (-deltas.where(deltas < 0, 0)).tail(rsi_period).mean()

            if loss == 0:
                rsi = 100
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

            return 1 if rsi < oversold else (-1 if rsi > overbought else 0)

        return StrategyInfo(
            name='Market Making',
            category='High Frequency',
            description=f'Market making with RSI({rsi_period})',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_order_flow_imbalance(
        volume_window: int = 10,
        volume_threshold: float = 0.1
    ) -> StrategyInfo:
        """Create order flow imbalance strategy"""
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < volume_window or 'volume' not in data.columns:
                # Fallback to price momentum if no volume
                if len(data) < volume_window:
                    return 0
                returns = data['close'].pct_change().tail(volume_window).mean()
                return 1 if returns > 0.001 else (-1 if returns < -0.001 else 0)

            volume_change = data['volume'].pct_change().tail(volume_window).mean()
            price_change = data['close'].pct_change().tail(volume_window).mean()

            if volume_change > volume_threshold:
                if price_change > 0:
                    return 1
                elif price_change < 0:
                    return -1
            return 0

        return StrategyInfo(
            name='Order Flow',
            category='High Frequency',
            description=f'Order flow imbalance with {volume_window}-period window',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_mean_variance(
        lookback: int = 60,
        rebalance_threshold: float = 0.05
    ) -> StrategyInfo:
        """Create mean-variance optimization strategy"""
        last_weight = [0.5]  # Start neutral

        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < lookback:
                return 0

            # Simple risk parity: buy when volatility is low, sell when high
            returns = data['close'].pct_change().tail(lookback)
            volatility = returns.std()

            # Calculate target weight (inverse volatility)
            avg_vol = 0.02  # Assumed average
            target_weight = min(1.0, avg_vol / (volatility + 1e-8))

            # Signal based on weight change
            weight_change = target_weight - last_weight[0]

            if abs(weight_change) > rebalance_threshold:
                last_weight[0] = target_weight
                return np.sign(weight_change)

            return 0

        return StrategyInfo(
            name='Mean Variance',
            category='Asset Allocation',
            description=f'Mean-variance optimization with {lookback}-period lookback',
            signal_generator=signal_func
        )

    @staticmethod
    def adapt_ml_strategy(
        strategy_type: str = 'random_forest',
        lookback: int = 20
    ) -> StrategyInfo:
        """
        Create ML-based strategy (simplified version using technical signals)

        In production, this would load a trained model and make predictions
        """
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < lookback:
                return 0

            # Feature engineering
            prices = data['close']
            returns = prices.pct_change()

            # Features
            momentum = returns.tail(lookback).mean()
            volatility = returns.tail(lookback).std()
            rsi = _calculate_rsi(prices, 14)

            # Simple rule-based prediction (proxy for ML)
            score = 0
            if momentum > 0.001:
                score += 1
            if momentum < -0.001:
                score -= 1
            if rsi < 30:
                score += 1
            if rsi > 70:
                score -= 1

            return np.sign(score) if score != 0 else 0

        return StrategyInfo(
            name=f'ML-{strategy_type.replace("_", " ").title()}',
            category='Machine Learning',
            description=f'{strategy_type.replace("_", " ").title()} strategy',
            signal_generator=signal_func
        )

    @staticmethod
    def get_all_strategies() -> Dict[str, StrategyInfo]:
        """Get all available strategies"""
        return {
            'Momentum': StrategyAdapter.adapt_momentum_strategy(),
            'Mean Reversion': StrategyAdapter.adapt_mean_reversion_strategy(),
            'Statistical Arbitrage': StrategyAdapter.adapt_statistical_arbitrage(),
            'Pairs Trading': StrategyAdapter.adapt_pairs_trading(),
            'Market Making': StrategyAdapter.adapt_market_making(),
            'Order Flow': StrategyAdapter.adapt_order_flow_imbalance(),
            'Mean Variance': StrategyAdapter.adapt_mean_variance(),
            'ML-Random Forest': StrategyAdapter.adapt_ml_strategy('random_forest'),
            'ML-XGBoost': StrategyAdapter.adapt_ml_strategy('xgboost'),
            'ML-LightGBM': StrategyAdapter.adapt_ml_strategy('lightgbm'),
        }


def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return 50  # Neutral

    deltas = prices.diff()
    gain = (deltas.where(deltas > 0, 0)).tail(period).mean()
    loss = (-deltas.where(deltas < 0, 0)).tail(period).mean()

    if loss == 0:
        return 100
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

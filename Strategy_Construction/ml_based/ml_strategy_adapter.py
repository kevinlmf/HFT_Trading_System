"""
Adapter to integrate ML strategies with the strategy comparison framework
Provides a unified interface for traditional and ML-based strategies
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

from strategy.ml_base import BaseMLStrategy, MLSignal


class TradingStrategyAdapter(ABC):
    """
    Abstract adapter for trading strategies
    Provides unified interface for both traditional and ML strategies
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate trading signals from price data"""
        pass

    @abstractmethod
    def calculate_position_sizes(
        self, signals: Dict[str, Any], portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate position sizes from signals"""
        pass


class MLStrategyAdapter(TradingStrategyAdapter):
    """
    Adapter for ML-based strategies
    Converts ML strategy outputs to traditional strategy interface
    """

    def __init__(
        self,
        ml_strategy: BaseMLStrategy,
        position_sizing: str = 'signal_strength',  # 'equal', 'signal_strength', 'confidence'
        max_position_size: float = 0.1,  # Max 10% per position
        min_confidence: float = 0.3
    ):
        super().__init__(name=ml_strategy.name)
        self.ml_strategy = ml_strategy
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence

    def generate_signals(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, MLSignal]:
        """Generate ML-based trading signals"""
        return self.ml_strategy.generate_signals(price_data)

    def calculate_position_sizes(
        self, signals: Dict[str, MLSignal], portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate position sizes from ML signals

        Args:
            signals: Dictionary of ML signals by symbol
            portfolio_value: Current portfolio value

        Returns:
            Dictionary of position weights by symbol (weight = fraction of portfolio)
        """
        positions = {}

        # Filter signals by confidence
        valid_signals = {
            symbol: signal for symbol, signal in signals.items()
            if signal.confidence >= self.min_confidence
        }

        if not valid_signals:
            return positions

        if self.position_sizing == 'equal':
            # Equal weight among valid signals
            n_positions = len(valid_signals)
            base_weight = min(self.max_position_size, 1.0 / n_positions)

            for symbol, signal in valid_signals.items():
                if signal.signal_strength > 0:  # Only long positions
                    positions[symbol] = base_weight

        elif self.position_sizing == 'signal_strength':
            # Weight by signal strength
            total_strength = sum(
                abs(signal.signal_strength) for signal in valid_signals.values()
                if signal.signal_strength > 0
            )

            if total_strength > 0:
                for symbol, signal in valid_signals.items():
                    if signal.signal_strength > 0:
                        weight = (
                            abs(signal.signal_strength) / total_strength
                        ) * min(0.95, self.max_position_size * len(valid_signals))
                        positions[symbol] = min(weight, self.max_position_size)

        elif self.position_sizing == 'confidence':
            # Weight by confidence
            total_confidence = sum(
                signal.confidence for signal in valid_signals.values()
                if signal.signal_strength > 0
            )

            if total_confidence > 0:
                for symbol, signal in valid_signals.items():
                    if signal.signal_strength > 0:
                        weight = (
                            signal.confidence / total_confidence
                        ) * min(0.95, self.max_position_size * len(valid_signals))
                        positions[symbol] = min(weight, self.max_position_size)

        # Normalize to ensure sum <= 1.0
        total_weight = sum(positions.values())
        if total_weight > 1.0:
            positions = {k: v / total_weight for k, v in positions.items()}

        return positions


class TraditionalStrategyAdapter(TradingStrategyAdapter):
    """
    Adapter for traditional strategies (momentum, pairs trading, mean-variance)
    Wraps existing strategy classes to provide unified interface
    """

    def __init__(self, strategy):
        super().__init__(name=strategy.__class__.__name__)
        self.strategy = strategy

    def generate_signals(self, price_data: Dict[str, pd.DataFrame]) -> Any:
        """Generate signals using traditional strategy"""
        return self.strategy.generate_signals(price_data)

    def calculate_position_sizes(
        self, signals: Any, portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate position sizes using traditional strategy"""
        return self.strategy.calculate_position_sizes(signals, portfolio_value)


class StrategySimulatorML:
    """
    Enhanced strategy simulator supporting both traditional and ML strategies
    """

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.initial_capital = float(initial_capital)

    def simulate_strategy(
        self,
        strategy: TradingStrategyAdapter,
        price_data: Dict[str, pd.DataFrame],
        rebalance_freq: int = 5,
        warmup_period: int = 60,
        transaction_cost: float = 0.001  # 0.1% per trade
    ) -> pd.Series:
        """
        Simulate trading strategy

        Args:
            strategy: Strategy adapter (traditional or ML)
            price_data: Dictionary of price DataFrames by symbol
            rebalance_freq: Rebalancing frequency in days
            warmup_period: Initial warm-up period to skip
            transaction_cost: Transaction cost as fraction of trade value

        Returns:
            Portfolio value series
        """
        portfolio_values = [self.initial_capital]
        previous_positions = {}

        # Get date index
        any_df = next(iter(price_data.values()))
        dates = any_df.index[warmup_period:]

        if len(dates) == 0:
            return pd.Series([self.initial_capital], index=any_df.index[:1])

        for date in dates[::rebalance_freq]:
            # Get current data up to date
            current_data = {sym: df.loc[:date] for sym, df in price_data.items()}

            # Generate signals
            signals = strategy.generate_signals(current_data)

            # Calculate position sizes
            positions = strategy.calculate_position_sizes(signals, portfolio_values[-1])

            # Calculate transaction costs
            tc = 0.0
            for sym, new_weight in positions.items():
                old_weight = previous_positions.get(sym, 0.0)
                turnover = abs(new_weight - old_weight)
                tc += turnover * transaction_cost

            # Calculate returns for this period
            daily_return = 0.0
            for sym, weight in positions.items():
                df = price_data.get(sym)
                if df is None or date not in df.index:
                    continue

                sub = df.loc[:date, "close"]
                if len(sub) < 2:
                    continue

                sym_ret = sub.pct_change().iloc[-1]
                if np.isfinite(sym_ret):
                    daily_return += float(weight) * float(sym_ret)

            # Apply transaction costs
            daily_return -= tc

            # Update portfolio
            portfolio_values.append(portfolio_values[-1] * (1.0 + daily_return))
            previous_positions = positions.copy()

        series = pd.Series(portfolio_values[1:], index=dates[::rebalance_freq], dtype=float)
        return series

    def backtest_ml_strategy(
        self,
        ml_strategy: BaseMLStrategy,
        price_data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Backtest ML strategy with training and out-of-sample testing

        Args:
            ml_strategy: ML strategy to backtest
            price_data: Price data for all symbols
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            **kwargs: Additional simulation parameters

        Returns:
            Tuple of (portfolio series, training info)
        """
        # Prepare data for training
        X_train, X_val, X_test, y_train, y_val, y_test = ml_strategy.prepare_data(
            price_data,
            train_split=train_ratio,
            val_split=val_ratio
        )

        # Train model
        print(f"Training {ml_strategy.name}...")
        training_result = ml_strategy.train(X_train, y_train, X_val, y_val)

        # Evaluate
        test_metrics = ml_strategy.evaluate(X_test, y_test)
        print(f"Test metrics: {test_metrics}")

        # Run simulation on full data
        adapter = MLStrategyAdapter(ml_strategy)
        portfolio_series = self.simulate_strategy(adapter, price_data, **kwargs)

        training_info = {
            'training_result': training_result,
            'test_metrics': test_metrics,
            'model_state': ml_strategy.state
        }

        return portfolio_series, training_info


# Convenience functions for creating adapters

def create_ml_adapter(
    strategy_class,
    prediction_type,
    **strategy_kwargs
) -> MLStrategyAdapter:
    """
    Create ML strategy adapter from strategy class

    Args:
        strategy_class: ML strategy class (e.g., RandomForestStrategy)
        prediction_type: Type of prediction (classification/regression)
        **strategy_kwargs: Additional strategy parameters

    Returns:
        MLStrategyAdapter instance
    """
    ml_strategy = strategy_class(prediction_type=prediction_type, **strategy_kwargs)
    return MLStrategyAdapter(ml_strategy)


def create_traditional_adapter(strategy_instance) -> TraditionalStrategyAdapter:
    """
    Create traditional strategy adapter

    Args:
        strategy_instance: Instance of traditional strategy

    Returns:
        TraditionalStrategyAdapter instance
    """
    return TraditionalStrategyAdapter(strategy_instance)

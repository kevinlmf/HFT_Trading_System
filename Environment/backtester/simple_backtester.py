"""
Simple Backtester for Strategy Comparison
Provides a unified interface to backtest any strategy regardless of implementation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0002  # 0.02% per trade
    slippage_bps: float = 1.0  # 1 basis point slippage
    position_size: float = 0.1  # 10% of capital per trade
    max_positions: int = 1  # Maximum concurrent positions


@dataclass
class BacktestResult:
    """Results from backtesting"""
    strategy_name: str
    returns: pd.Series  # Daily returns
    trades: pd.DataFrame  # Trade history
    equity_curve: pd.Series  # Portfolio value over time
    positions: pd.DataFrame  # Position history
    metrics: Dict[str, float]  # Performance metrics
    config: BacktestConfig


class SimpleBacktester:
    """
    Simple backtester that can run any strategy

    Works with strategies that have various interfaces:
    - generate_signal(data) -> signal
    - decide_position(prices, portfolio_value) -> positions
    - calculate_quotes(mid_price) -> (bid, ask)
    - etc.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy_name: str,
        strategy_func: Callable,
        data: pd.DataFrame,
        signal_column: str = 'signal',
        price_column: str = 'close'
    ) -> BacktestResult:
        """
        Run backtest on a strategy

        Args:
            strategy_name: Name of the strategy
            strategy_func: Function that generates signals from data
                          Should accept DataFrame and return signals
            data: Historical market data with OHLCV
            signal_column: Column name for signals in output
            price_column: Column name for prices

        Returns:
            BacktestResult with complete results
        """
        # Initialize
        capital = self.config.initial_capital
        position = 0
        cash = capital

        trades = []
        equity_curve = []
        positions_history = []

        # Run through data
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_price = data[price_column].iloc[i]
            current_time = data.index[i]

            # Skip if not enough data
            if i < 20:  # Minimum lookback
                equity_curve.append(capital)
                positions_history.append({
                    'timestamp': current_time,
                    'position': position,
                    'cash': cash,
                    'equity': capital
                })
                continue

            # Generate signal
            try:
                signal = strategy_func(current_data)
                # 如果返回Series，取最后一个值
                if isinstance(signal, pd.Series):
                    if len(signal) > 0:
                        signal = signal.iloc[-1]
                    else:
                        signal = 0
                # 确保是数值类型
                if pd.isna(signal):
                    signal = 0
                signal = float(signal) if signal is not None else 0
            except Exception as e:
                import warnings
                warnings.warn(f"Strategy signal generation failed at step {i}: {e}")
                signal = 0

            # Execute trades based on signal
            if signal > 0 and position <= 0:  # Buy signal
                # Close short if any
                if position < 0:
                    pnl = (-position) * (current_price - abs(position) / (-position))
                    cash += abs(position) * current_price
                    commission = abs(position) * current_price * self.config.commission_rate
                    slippage = abs(position) * current_price * (self.config.slippage_bps / 10000)
                    cash -= (commission + slippage)

                    trades.append({
                        'timestamp': current_time,
                        'action': 'COVER',
                        'price': current_price,
                        'quantity': abs(position),
                        'pnl': pnl,
                        'commission': commission,
                        'slippage': slippage
                    })
                    position = 0

                # Open long
                position_value = cash * self.config.position_size
                quantity = position_value / current_price
                commission = position_value * self.config.commission_rate
                slippage = position_value * (self.config.slippage_bps / 10000)

                if cash >= (position_value + commission + slippage):
                    position = quantity
                    cash -= (position_value + commission + slippage)

                    trades.append({
                        'timestamp': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': quantity,
                        'pnl': 0,
                        'commission': commission,
                        'slippage': slippage
                    })

            elif signal < 0 and position >= 0:  # Sell signal
                # Close long if any
                if position > 0:
                    pnl = position * current_price - (cash - (capital - position * current_price))
                    cash += position * current_price
                    commission = position * current_price * self.config.commission_rate
                    slippage = position * current_price * (self.config.slippage_bps / 10000)
                    cash -= (commission + slippage)

                    trades.append({
                        'timestamp': current_time,
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': position,
                        'pnl': pnl,
                        'commission': commission,
                        'slippage': slippage
                    })
                    position = 0

            # Calculate equity
            if position > 0:
                equity = cash + position * current_price
            elif position < 0:
                equity = cash - abs(position) * current_price
            else:
                equity = cash

            equity_curve.append(equity)
            positions_history.append({
                'timestamp': current_time,
                'position': position,
                'cash': cash,
                'equity': equity,
                'price': current_price
            })

        # Convert to DataFrames/Series
        equity_series = pd.Series(equity_curve, index=data.index)
        returns = equity_series.pct_change().fillna(0)

        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df['symbol'] = strategy_name
            trades_df['strategy'] = strategy_name
            # Add duration (dummy for now)
            trades_df['duration'] = 1
        else:
            trades_df = pd.DataFrame(columns=['timestamp', 'action', 'price', 'quantity',
                                             'pnl', 'commission', 'slippage', 'symbol',
                                             'strategy', 'duration'])

        positions_df = pd.DataFrame(positions_history)

        # Calculate basic metrics
        metrics = self._calculate_metrics(returns, trades_df, equity_series)

        return BacktestResult(
            strategy_name=strategy_name,
            returns=returns,
            trades=trades_df,
            equity_curve=equity_series,
            positions=positions_df,
            metrics=metrics,
            config=self.config
        )

    def _calculate_metrics(
        self,
        returns: pd.Series,
        trades: pd.DataFrame,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return (assuming daily data)
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        # Sharpe ratio
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        # Max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        if len(trades) > 0 and 'pnl' in trades.columns:
            winning_trades = (trades['pnl'] > 0).sum()
            win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'final_equity': equity_curve.iloc[-1]
        }


def create_signal_generator(strategy_instance, strategy_type: str) -> Callable:
    """
    Create a signal generator function from a strategy instance

    Args:
        strategy_instance: Instance of any strategy class
        strategy_type: Type of strategy ('momentum', 'mean_reversion', 'market_making', etc.)

    Returns:
        Function that generates signals from data
    """

    if strategy_type == 'momentum':
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < 20:
                return 0
            returns = data['close'].pct_change()
            momentum = returns.tail(20).mean()
            return 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)
        return signal_func

    elif strategy_type == 'mean_reversion':
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < 20:
                return 0
            prices = data['close']
            mean = prices.tail(20).mean()
            std = prices.tail(20).std()
            current = prices.iloc[-1]
            z_score = (current - mean) / std if std > 0 else 0
            return -1 if z_score > 2 else (1 if z_score < -2 else 0)
        return signal_func

    elif strategy_type == 'stat_arb':
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < 60:
                return 0
            prices = data['close']
            sma_20 = prices.tail(20).mean()
            sma_60 = prices.tail(60).mean()
            return 1 if sma_20 > sma_60 else -1
        return signal_func

    elif strategy_type == 'market_making':
        # Market making doesn't fit this framework well
        # Use a simple oscillator instead
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < 20:
                return 0
            prices = data['close']
            rsi = _calculate_rsi(prices, 14)
            return 1 if rsi < 30 else (-1 if rsi > 70 else 0)
        return signal_func

    elif strategy_type == 'order_flow':
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < 10:
                return 0
            if 'volume' in data.columns:
                volume_change = data['volume'].pct_change().tail(10).mean()
                price_change = data['close'].pct_change().tail(10).mean()
                if volume_change > 0.1 and price_change > 0:
                    return 1
                elif volume_change > 0.1 and price_change < 0:
                    return -1
            return 0
        return signal_func

    else:
        # Default: simple momentum
        def signal_func(data: pd.DataFrame) -> float:
            if len(data) < 20:
                return 0
            returns = data['close'].pct_change().tail(20).mean()
            return 1 if returns > 0 else -1
        return signal_func


def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI indicator"""
    deltas = prices.diff()
    gain = (deltas.where(deltas > 0, 0)).tail(period).mean()
    loss = (-deltas.where(deltas < 0, 0)).tail(period).mean()

    if loss == 0:
        return 100
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def run_backtest(symbols: List[str] = None, capital: float = 100000.0):
    """
    Convenience function to run backtests on multiple strategies

    Args:
        symbols: List of symbols to backtest (default: ['SPY'])
        capital: Initial capital (default: 100000)
    """
    if symbols is None:
        symbols = ['SPY']

    print("=" * 80)
    print("BACKTESTING ENGINE")
    print("=" * 80)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Initial Capital: ${capital:,.2f}\n")

    # Generate sample data for demonstration
    # In production, you would load real historical data
    print("Generating sample data...")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1h')

    # Create synthetic price data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()

    data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)

    print(f"Generated {len(data)} data points\n")

    # Define strategies to test
    strategies_to_test = [
        ('momentum', 'Momentum Strategy'),
        ('mean_reversion', 'Mean Reversion Strategy'),
        ('stat_arb', 'Statistical Arbitrage'),
    ]

    # Configure backtester
    config = BacktestConfig(
        initial_capital=capital,
        commission_rate=0.0002,
        slippage_bps=1.0,
        position_size=0.1,
        max_positions=1
    )

    backtester = SimpleBacktester(config)

    # Run backtests
    results = {}
    print("Running backtests...")
    print("-" * 80)

    for strategy_type, strategy_name in strategies_to_test:
        print(f"\nTesting: {strategy_name}")

        # Create signal generator for this strategy
        signal_func = create_signal_generator(None, strategy_type)

        # Run backtest
        result = backtester.run(
            strategy_name=strategy_name,
            strategy_func=signal_func,
            data=data,
            price_column='close'
        )

        results[strategy_name] = result

        # Print metrics
        metrics = result.metrics
        print(f"  Total Return:      {metrics['total_return']*100:>8.2f}%")
        print(f"  Annual Return:     {metrics['annualized_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Win Rate:          {metrics['win_rate']*100:>8.2f}%")
        print(f"  Number of Trades:  {metrics['num_trades']:>8.0f}")
        print(f"  Final Equity:      ${metrics['final_equity']:>12,.2f}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)

    for name, result in results.items():
        m = result.metrics
        print(f"{name:<30} {m['total_return']*100:>10.2f}% {m['sharpe_ratio']:>9.2f} "
              f"{m['max_drawdown']*100:>10.2f}% {m['num_trades']:>9.0f}")

    print("\n" + "=" * 80)
    print("Backtest complete!")
    print("=" * 80)

    return results

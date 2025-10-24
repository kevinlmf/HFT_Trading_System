"""
Quick Strategy Comparison Script
Rapidly compares all available strategies and generates a summary report
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from execution.evaluation import StrategyEvaluator
from execution.engine.simple_backtester import SimpleBacktester, BacktestConfig
from execution.engine.strategy_adapter import StrategyAdapter


def generate_market_data(n_days: int = 504, volatility: float = 0.015) -> pd.DataFrame:
    """Generate sample market data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    returns = np.random.randn(n_days) * volatility + 0.0003
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1) * (1 + np.random.randn(n_days) * volatility * 0.1)
    data['open'].iloc[0] = 100
    daily_range = prices * volatility * np.random.uniform(0.5, 2.0, n_days)
    data['high'] = prices + daily_range * np.random.uniform(0, 1, n_days)
    data['low'] = prices - daily_range * np.random.uniform(0, 1, n_days)
    base_volume = 1_000_000
    volume_pattern = np.sin(np.arange(n_days) * 2 * np.pi / 20) * 0.3 + 1
    data['volume'] = (base_volume * volume_pattern * (1 + np.random.randn(n_days) * 0.2)).astype(int)

    return data


def main():
    """Quick comparison of all strategies"""

    print("\n" + "=" * 100)
    print("QUICK STRATEGY COMPARISON - ALL STRATEGIES".center(100))
    print("=" * 100)

    # Generate data
    print("\n📊 Generating market data (2 years)...")
    data = generate_market_data()
    print(f"✓ Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

    # Get strategies
    strategies = StrategyAdapter.get_all_strategies()
    print(f"\n🎯 Testing {len(strategies)} strategies...")

    # Initialize backtester
    backtester = SimpleBacktester(BacktestConfig(
        initial_capital=1_000_000.0,
        commission_rate=0.0002,
        slippage_bps=1.0,
        position_size=0.1
    ))

    # Run backtests
    results = []
    print("\n" + "-" * 100)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8} {'Win Rate':>10}")
    print("-" * 100)

    for name, info in strategies.items():
        try:
            result = backtester.run(name, info.signal_generator, data)

            results.append({
                'Strategy': name,
                'Category': info.category,
                'Total Return': result.metrics['total_return'],
                'Annual Return': result.metrics['annualized_return'],
                'Sharpe Ratio': result.metrics['sharpe_ratio'],
                'Max Drawdown': result.metrics['max_drawdown'],
                'Num Trades': result.metrics['num_trades'],
                'Win Rate': result.metrics['win_rate'],
                'Final Equity': result.metrics['final_equity']
            })

            print(f"{name:<25} {result.metrics['total_return']:>9.2%} "
                  f"{result.metrics['sharpe_ratio']:>8.2f} "
                  f"{result.metrics['max_drawdown']:>9.2%} "
                  f"{result.metrics['num_trades']:>8.0f} "
                  f"{result.metrics['win_rate']:>9.2%}")

        except Exception as e:
            print(f"{name:<25} {'ERROR':>10} {str(e)[:50]}")
            continue

    print("-" * 100)

    # Create results DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("\n⚠️  No strategies completed successfully!")
        return

    # Rankings
    print("\n" + "=" * 100)
    print("RANKINGS".center(100))
    print("=" * 100)

    # By Sharpe Ratio
    print("\n🏆 Top 5 by Sharpe Ratio:")
    top_sharpe = df.nlargest(5, 'Sharpe Ratio')[['Strategy', 'Sharpe Ratio', 'Annual Return', 'Max Drawdown']]
    for idx, row in top_sharpe.iterrows():
        print(f"  {idx+1}. {row['Strategy']:<25} Sharpe={row['Sharpe Ratio']:>6.2f}  "
              f"Return={row['Annual Return']:>7.2%}  DD={row['Max Drawdown']:>7.2%}")

    # By Return
    print("\n📈 Top 5 by Annual Return:")
    top_return = df.nlargest(5, 'Annual Return')[['Strategy', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']]
    for idx, row in top_return.iterrows():
        print(f"  {idx+1}. {row['Strategy']:<25} Return={row['Annual Return']:>7.2%}  "
              f"Sharpe={row['Sharpe Ratio']:>6.2f}  DD={row['Max Drawdown']:>7.2%}")

    # By Risk (lowest drawdown)
    print("\n🛡️  Top 5 by Lowest Risk (Max Drawdown):")
    top_risk = df.nsmallest(5, 'Max Drawdown', keep='first')[['Strategy', 'Max Drawdown', 'Sharpe Ratio', 'Annual Return']]
    for idx, row in top_risk.iterrows():
        print(f"  {idx+1}. {row['Strategy']:<25} DD={row['Max Drawdown']:>7.2%}  "
              f"Sharpe={row['Sharpe Ratio']:>6.2f}  Return={row['Annual Return']:>7.2%}")

    # By Category
    print("\n📊 Performance by Category:")
    category_stats = df.groupby('Category').agg({
        'Sharpe Ratio': 'mean',
        'Annual Return': 'mean',
        'Max Drawdown': 'mean',
        'Win Rate': 'mean'
    }).round(4)
    print(category_stats.to_string())

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS".center(100))
    print("=" * 100)

    print(f"\nTotal Strategies Tested: {len(df)}")
    print(f"\nSharpe Ratio:    Mean={df['Sharpe Ratio'].mean():>7.2f}  "
          f"Median={df['Sharpe Ratio'].median():>7.2f}  "
          f"Best={df['Sharpe Ratio'].max():>7.2f}")
    print(f"Annual Return:   Mean={df['Annual Return'].mean():>7.2%}  "
          f"Median={df['Annual Return'].median():>7.2%}  "
          f"Best={df['Annual Return'].max():>7.2%}")
    print(f"Max Drawdown:    Mean={df['Max Drawdown'].mean():>7.2%}  "
          f"Median={df['Max Drawdown'].median():>7.2%}  "
          f"Best={df['Max Drawdown'].max():>7.2%}")
    print(f"Win Rate:        Mean={df['Win Rate'].mean():>7.2%}  "
          f"Median={df['Win Rate'].median():>7.2%}  "
          f"Best={df['Win Rate'].max():>7.2%}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS".center(100))
    print("=" * 100)

    # Best overall (combined metric)
    df['Combined Score'] = (
        df['Sharpe Ratio'].rank(pct=True) * 0.4 +
        df['Annual Return'].rank(pct=True) * 0.3 +
        (1 - df['Max Drawdown'].abs().rank(pct=True)) * 0.3
    )

    best_overall = df.nlargest(1, 'Combined Score').iloc[0]
    print(f"\n🌟 BEST OVERALL STRATEGY: {best_overall['Strategy']}")
    print(f"   Category: {best_overall['Category']}")
    print(f"   Sharpe Ratio: {best_overall['Sharpe Ratio']:.2f}")
    print(f"   Annual Return: {best_overall['Annual Return']:.2%}")
    print(f"   Max Drawdown: {best_overall['Max Drawdown']:.2%}")
    print(f"   Win Rate: {best_overall['Win Rate']:.2%}")
    print(f"   Number of Trades: {best_overall['Num Trades']:.0f}")

    # Conservative choice
    df['Conservative Score'] = (
        (1 - df['Max Drawdown'].abs().rank(pct=True)) * 0.5 +
        df['Sharpe Ratio'].rank(pct=True) * 0.5
    )
    best_conservative = df.nlargest(1, 'Conservative Score').iloc[0]
    print(f"\n🛡️  BEST CONSERVATIVE STRATEGY: {best_conservative['Strategy']}")
    print(f"   Sharpe Ratio: {best_conservative['Sharpe Ratio']:.2f}")
    print(f"   Annual Return: {best_conservative['Annual Return']:.2%}")
    print(f"   Max Drawdown: {best_conservative['Max Drawdown']:.2%}")

    # Aggressive choice
    df['Aggressive Score'] = (
        df['Annual Return'].rank(pct=True) * 0.7 +
        df['Sharpe Ratio'].rank(pct=True) * 0.3
    )
    best_aggressive = df.nlargest(1, 'Aggressive Score').iloc[0]
    print(f"\n🚀 BEST AGGRESSIVE STRATEGY: {best_aggressive['Strategy']}")
    print(f"   Annual Return: {best_aggressive['Annual Return']:.2%}")
    print(f"   Sharpe Ratio: {best_aggressive['Sharpe Ratio']:.2f}")
    print(f"   Max Drawdown: {best_aggressive['Max Drawdown']:.2%}")

    # Save results
    output_file = 'strategy_comparison_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

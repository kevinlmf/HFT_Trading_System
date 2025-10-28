"""
Demo Script: Machine Learning Trading Strategies
Demonstrates how to use and compare ML-based trading strategies
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.ml_base import PredictionType, FeatureEngineer
from strategy.ml_traditional import RandomForestStrategy, XGBoostStrategy, LightGBMStrategy
from strategy.ml_strategy_adapter import MLStrategyAdapter, StrategySimulatorML


def generate_demo_data(n_days: int = 500) -> dict[str, pd.DataFrame]:
    """Generate sample price data for demonstration"""
    print("Generating sample price data...")

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data = {}

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    for symbol in symbols:
        # Generate random walk with drift
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.randn(n_days) * 0.02 + 0.0005
        prices = 100 * np.cumprod(1 + returns)

        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.01, n_days),
            'high': prices * np.random.uniform(1.00, 1.02, n_days),
            'low': prices * np.random.uniform(0.98, 1.00, n_days),
            'close': prices,
            'volume': np.random.randint(1_000_000, 10_000_000, n_days)
        }, index=dates)

        data[symbol] = df

    return data


def demo_single_ml_strategy():
    """Demo: Train and test a single ML strategy"""
    print("\n" + "="*80)
    print("DEMO 1: Single ML Strategy (Random Forest)")
    print("="*80)

    # Generate data
    price_data = generate_demo_data(n_days=500)

    # Create Random Forest strategy
    rf_strategy = RandomForestStrategy(
        prediction_type=PredictionType.CLASSIFICATION,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    print("\nPreparing data for training...")
    X_train, X_val, X_test, y_train, y_val, y_test = rf_strategy.prepare_data(
        price_data,
        train_split=0.6,
        val_split=0.2
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    print("\nTraining Random Forest model...")
    training_result = rf_strategy.train(X_train, y_train, X_val, y_val)

    print("\nTraining Results:")
    print(f"  Train Score: {training_result['train_score']:.4f}")
    print(f"  Validation Score: {training_result['val_score']:.4f}")

    print("\nEvaluating on test set...")
    test_metrics = rf_strategy.evaluate(X_test, y_test)

    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTop 10 Most Important Features:")
    sorted_features = sorted(
        training_result['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")

    return rf_strategy, price_data


def demo_ml_strategy_comparison():
    """Demo: Compare multiple ML strategies"""
    print("\n" + "="*80)
    print("DEMO 2: ML Strategy Comparison")
    print("="*80)

    # Generate data
    price_data = generate_demo_data(n_days=500)

    # Create strategies
    strategies = {
        'Random Forest': RandomForestStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=50,
            max_depth=8,
            random_state=42
        ),
        'XGBoost': XGBoostStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'LightGBM': LightGBMStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"\nTraining {name}...")

        try:
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = strategy.prepare_data(
                price_data,
                train_split=0.6,
                val_split=0.2
            )

            # Train
            training_result = strategy.train(X_train, y_train, X_val, y_val)

            # Evaluate
            test_metrics = strategy.evaluate(X_test, y_test)

            results[name] = {
                'train_score': training_result.get('train_score', 0),
                'val_score': training_result.get('val_score', 0),
                'test_metrics': test_metrics
            }

            print(f"  ✓ {name} completed")
            print(f"    Validation Score: {training_result.get('val_score', 0):.4f}")
            print(f"    Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")

        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results[name] = None

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Strategy':<20} {'Val Score':<12} {'Test Acc':<12} {'Precision':<12} {'F1 Score':<12}")
    print("-"*80)

    for name, result in results.items():
        if result is not None:
            val_score = result['val_score']
            test_acc = result['test_metrics'].get('accuracy', 0)
            precision = result['test_metrics'].get('precision', 0)
            f1 = result['test_metrics'].get('f1', 0)

            print(f"{name:<20} {val_score:<12.4f} {test_acc:<12.4f} {precision:<12.4f} {f1:<12.4f}")
        else:
            print(f"{name:<20} {'FAILED':<12}")

    return results


def demo_backtesting():
    """Demo: Backtest ML strategy"""
    print("\n" + "="*80)
    print("DEMO 3: ML Strategy Backtesting")
    print("="*80)

    # Generate data
    price_data = generate_demo_data(n_days=500)

    # Create strategy
    strategy = XGBoostStrategy(
        prediction_type=PredictionType.CLASSIFICATION,
        n_estimators=50,
        max_depth=6,
        random_state=42
    )

    # Create simulator
    simulator = StrategySimulatorML(initial_capital=1_000_000.0)

    print("\nRunning backtest with training...")
    portfolio_series, training_info = simulator.backtest_ml_strategy(
        strategy,
        price_data,
        train_ratio=0.6,
        val_ratio=0.2,
        rebalance_freq=5,
        warmup_period=100
    )

    # Calculate performance metrics
    returns = portfolio_series.pct_change().dropna()

    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_series)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

    max_dd = ((portfolio_series / portfolio_series.cummax()) - 1).min()

    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Initial Capital:    ${simulator.initial_capital:,.2f}")
    print(f"Final Value:        ${portfolio_series.iloc[-1]:,.2f}")
    print(f"Total Return:       {total_return:.2%}")
    print(f"Annual Return:      {annual_return:.2%}")
    print(f"Volatility:         {volatility:.2%}")
    print(f"Sharpe Ratio:       {sharpe:.3f}")
    print(f"Max Drawdown:       {max_dd:.2%}")

    print("\nModel Performance:")
    test_metrics = training_info['test_metrics']
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return portfolio_series, training_info


def demo_feature_engineering():
    """Demo: Feature engineering for ML strategies"""
    print("\n" + "="*80)
    print("DEMO 4: Feature Engineering")
    print("="*80)

    # Generate simple price data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    prices = 100 * np.cumprod(1 + np.random.randn(200) * 0.02)

    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.99, 1.01, 200),
        'high': prices * np.random.uniform(1.00, 1.02, 200),
        'low': prices * np.random.uniform(0.98, 1.00, 200),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, 200)
    }, index=dates)

    print("\nOriginal data shape:", df.shape)
    print("\nOriginal columns:")
    print(df.columns.tolist())

    # Create feature engineer
    feature_engineer = FeatureEngineer(lookback_periods=[5, 10, 20])

    print("\nGenerating features...")
    features_df = feature_engineer.generate_features(df, symbol='DEMO')

    print(f"\nAfter feature engineering: {features_df.shape}")
    print(f"Number of new features: {len(feature_engineer.get_feature_names())}")

    print("\nSample of generated features:")
    feature_names = feature_engineer.get_feature_names()
    print(f"Total features: {len(feature_names)}")

    # Group features by type
    feature_types = {
        'Price': [f for f in feature_names if 'return' in f or 'momentum' in f or 'price_position' in f],
        'Technical': [f for f in feature_names if any(x in f for x in ['sma', 'ema', 'rsi', 'macd', 'bb'])],
        'Volume': [f for f in feature_names if 'volume' in f or 'obv' in f or 'vwap' in f or 'mfi' in f],
        'Microstructure': [f for f in feature_names if 'volatility' in f or 'spread' in f or 'impact' in f],
        'Time': [f for f in feature_names if any(x in f for x in ['day', 'month', 'quarter', 'sin', 'cos'])],
        'Lag': [f for f in feature_names if 'lag' in f]
    }

    print("\nFeatures by type:")
    for feature_type, features in feature_types.items():
        print(f"  {feature_type}: {len(features)} features")
        if len(features) > 0:
            print(f"    Examples: {', '.join(features[:3])}")


def main():
    """Main demo function"""
    print("="*80)
    print("MACHINE LEARNING TRADING STRATEGIES DEMO")
    print("="*80)
    print("\nThis demo showcases:")
    print("  1. Training a single ML strategy")
    print("  2. Comparing multiple ML strategies")
    print("  3. Backtesting ML strategies")
    print("  4. Feature engineering for ML")
    print()

    try:
        # Demo 1: Single strategy
        demo_single_ml_strategy()

        # Demo 2: Strategy comparison
        demo_ml_strategy_comparison()

        # Demo 3: Backtesting
        demo_backtesting()

        # Demo 4: Feature engineering
        demo_feature_engineering()

        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run full strategy comparison: python scripts/ml_strategy_comparison.py")
        print("  2. Experiment with different ML models and hyperparameters")
        print("  3. Try deep learning strategies (LSTM, Transformer)")
        print("  4. Explore reinforcement learning strategies")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

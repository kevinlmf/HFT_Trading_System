"""
Example script demonstrating ML/DL trading strategies
Shows how to train and evaluate different strategy types
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt

# Import strategies
from strategy.ml_base import PredictionType, FeatureEngineer
from strategy.ml_traditional import (
    RandomForestStrategy,
    XGBoostStrategy,
    LightGBMStrategy,
    GradientBoostingStrategy,
    EnsembleMLStrategy
)
from strategy.dl_strategies import LSTMStrategy, TransformerStrategy, CNNStrategy
from strategy.rl_strategies import RLTradingStrategy

# Import environment
from env.advanced_trading_env import AdvancedTradingEnv


def generate_sample_data(symbols: list = ['AAPL', 'GOOGL', 'MSFT'],
                         n_days: int = 500) -> Dict[str, pd.DataFrame]:
    """Generate sample price data for demonstration"""
    data_dict = {}

    for symbol in symbols:
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
        np.random.seed(hash(symbol) % 2**32)

        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'high': [p * np.random.uniform(1.00, 1.03) for p in prices],
            'low': [p * np.random.uniform(0.97, 1.00) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)

        data_dict[symbol] = df

    return data_dict


def example_traditional_ml():
    """Example: Train and evaluate traditional ML strategies"""
    print("\n" + "="*80)
    print("TRADITIONAL MACHINE LEARNING STRATEGIES")
    print("="*80)

    # Generate sample data
    print("\n1. Generating sample data...")
    price_data = generate_sample_data()

    # Initialize strategies
    strategies = {
        'RandomForest': RandomForestStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=100,
            max_depth=10
        ),
        'XGBoost': XGBoostStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        'LightGBM': LightGBMStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=100,
            num_leaves=31
        ),
    }

    # Train each strategy
    results = {}

    for name, strategy in strategies.items():
        print(f"\n2. Training {name}...")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = strategy.prepare_data(
            price_data,
            train_split=0.7,
            val_split=0.15
        )

        # Train
        training_result = strategy.train(X_train, y_train, X_val, y_val)
        print(f"   Training complete!")
        print(f"   Train score: {training_result.get('train_score', 'N/A')}")
        print(f"   Val score: {training_result.get('val_score', 'N/A')}")

        # Evaluate
        print(f"\n3. Evaluating {name}...")
        eval_metrics = strategy.evaluate(X_test, y_test)
        results[name] = eval_metrics

        print(f"   Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {eval_metrics.get('precision', 0):.4f}")
        print(f"   Recall: {eval_metrics.get('recall', 0):.4f}")
        print(f"   F1: {eval_metrics.get('f1', 0):.4f}")

        # Show top features
        print(f"\n4. Top 10 important features for {name}:")
        feature_importance = sorted(
            strategy.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for i, (feature, importance) in enumerate(feature_importance, 1):
            print(f"   {i}. {feature}: {importance:.4f}")

        # Generate signals
        print(f"\n5. Generating trading signals with {name}...")
        signals = strategy.generate_signals(price_data)

        for symbol, signal in signals.items():
            print(f"   {symbol}: Strength={signal.signal_strength:.3f}, "
                  f"Confidence={signal.confidence:.3f}")

    return results, strategies


def example_ensemble_ml():
    """Example: Ensemble ML strategy"""
    print("\n" + "="*80)
    print("ENSEMBLE ML STRATEGY")
    print("="*80)

    # Generate sample data
    print("\n1. Generating sample data...")
    price_data = generate_sample_data()

    # Create base strategies
    base_strategies = [
        RandomForestStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=50
        ),
        XGBoostStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=50
        ),
        LightGBMStrategy(
            prediction_type=PredictionType.CLASSIFICATION,
            n_estimators=50
        )
    ]

    # Create ensemble
    print("\n2. Creating ensemble strategy...")
    ensemble = EnsembleMLStrategy(
        strategies=base_strategies,
        weights=[0.4, 0.3, 0.3],
        voting_method='soft'
    )

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = ensemble.prepare_data(
        price_data,
        train_split=0.7,
        val_split=0.15
    )

    # Train ensemble
    print("\n3. Training ensemble...")
    training_results = ensemble.train(X_train, y_train, X_val, y_val)

    for strategy_name, result in training_results.items():
        print(f"   {strategy_name}: Train score={result.get('train_score', 'N/A')}")

    # Evaluate ensemble
    print("\n4. Evaluating ensemble...")
    eval_metrics = ensemble.evaluate(X_test, y_test)

    print(f"   Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
    print(f"   Precision: {eval_metrics.get('precision', 0):.4f}")
    print(f"   Recall: {eval_metrics.get('recall', 0):.4f}")
    print(f"   F1: {eval_metrics.get('f1', 0):.4f}")

    return ensemble


def example_deep_learning():
    """Example: Deep learning strategies"""
    print("\n" + "="*80)
    print("DEEP LEARNING STRATEGIES")
    print("="*80)

    try:
        import torch

        # Generate sample data
        print("\n1. Generating sample data...")
        price_data = generate_sample_data()

        # Initialize DL strategies
        strategies = {
            'LSTM': LSTMStrategy(
                prediction_type=PredictionType.CLASSIFICATION,
                hidden_size=64,
                num_layers=2,
                epochs=20,
                batch_size=32
            ),
            'Transformer': TransformerStrategy(
                prediction_type=PredictionType.CLASSIFICATION,
                d_model=64,
                nhead=4,
                num_layers=2,
                epochs=20,
                batch_size=32
            ),
            'CNN': CNNStrategy(
                prediction_type=PredictionType.CLASSIFICATION,
                num_filters=[32, 64, 128],
                epochs=20,
                batch_size=32
            )
        }

        results = {}

        for name, strategy in strategies.items():
            print(f"\n2. Training {name}...")

            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = strategy.prepare_data(
                price_data,
                train_split=0.7,
                val_split=0.15
            )

            # Train
            training_result = strategy.train(X_train, y_train, X_val, y_val)
            print(f"   Training complete!")

            # Evaluate
            print(f"\n3. Evaluating {name}...")
            eval_metrics = strategy.evaluate(X_test, y_test)
            results[name] = eval_metrics

            print(f"   Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
            print(f"   Precision: {eval_metrics.get('precision', 0):.4f}")
            print(f"   F1: {eval_metrics.get('f1', 0):.4f}")

        return results, strategies

    except ImportError:
        print("\nPyTorch not available. Skipping deep learning examples.")
        return None, None


def example_reinforcement_learning():
    """Example: Reinforcement learning strategy"""
    print("\n" + "="*80)
    print("REINFORCEMENT LEARNING STRATEGY")
    print("="*80)

    try:
        import torch

        # Create trading environment
        print("\n1. Creating trading environment...")
        env = AdvancedTradingEnv(
            symbols='AAPL',
            initial_balance=100000,
            window_size=20
        )

        # Create RL strategy
        print("\n2. Initializing RL strategy (DQN)...")

        # Get state and action dimensions
        obs, _ = env.reset()
        state_dim = obs.flatten().shape[0]
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

        rl_strategy = RLTradingStrategy(
            agent_type='DQN',
            state_dim=state_dim,
            action_dim=action_dim,
            use_stable_baselines=False,  # Use custom implementation
            learning_rate=0.001,
            gamma=0.99,
            batch_size=64
        )

        # Train
        print("\n3. Training RL agent (this may take a while)...")
        episode_rewards = rl_strategy.train_on_environment(
            env,
            n_episodes=100,  # Reduced for demo
            max_steps=500
        )

        print(f"\n4. Training complete!")
        print(f"   Average reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
        print(f"   Best episode reward: {np.max(episode_rewards):.2f}")

        # Test the trained agent
        print("\n5. Testing trained agent...")
        obs, _ = env.reset()
        total_reward = 0

        for step in range(100):
            action = rl_strategy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        print(f"   Test episode reward: {total_reward:.2f}")
        print(f"   Final portfolio value: ${info['portfolio_value']:.2f}")
        print(f"   Total return: {info['total_return']:.2%}")

        return rl_strategy, episode_rewards

    except ImportError as e:
        print(f"\nRequired library not available: {e}")
        print("Skipping reinforcement learning example.")
        return None, None


def plot_comparison(results: Dict[str, Dict[str, float]], title: str = "Strategy Comparison"):
    """Plot comparison of strategy performances"""
    try:
        import matplotlib.pyplot as plt

        strategies = list(results.keys())
        metrics = list(results[strategies[0]].keys())

        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            values = [results[strategy][metric] for strategy in strategies]
            axes[i].bar(strategies, values)
            axes[i].set_title(metric.upper())
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to 'strategy_comparison.png'")

    except ImportError:
        print("\nMatplotlib not available. Skipping plot generation.")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ML/DL TRADING STRATEGIES - COMPREHENSIVE EXAMPLES")
    print("="*80)

    # Traditional ML
    try:
        ml_results, ml_strategies = example_traditional_ml()
        plot_comparison(ml_results, "Traditional ML Strategies Comparison")
    except Exception as e:
        print(f"\nError in traditional ML example: {e}")

    # Ensemble ML
    try:
        ensemble = example_ensemble_ml()
    except Exception as e:
        print(f"\nError in ensemble ML example: {e}")

    # Deep Learning
    try:
        dl_results, dl_strategies = example_deep_learning()
        if dl_results:
            plot_comparison(dl_results, "Deep Learning Strategies Comparison")
    except Exception as e:
        print(f"\nError in deep learning example: {e}")

    # Reinforcement Learning
    try:
        rl_strategy, rl_rewards = example_reinforcement_learning()
    except Exception as e:
        print(f"\nError in reinforcement learning example: {e}")

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()

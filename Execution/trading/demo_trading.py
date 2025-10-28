#!/usr/bin/env python3
"""
Demo Real-Time Trading System

Simple example to get started with real-time trading.
This uses paper trading (no real money) for testing.

Requirements:
1. Free Alpaca account: https://alpaca.markets
2. Get API keys from dashboard
3. Run: python demo_trading.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Data.connectors.alpaca_connector import AlpacaConnector
from Data.connectors.base_connector import MarketTick
from Execution.trading.strategy_router.adaptive_router import AdaptiveStrategyRouter
from Monitoring.pnl_tracking.realtime_pnl import RealTimePnLTracker

# Simple mock strategies for demo
class SimpleStrategy:
    """Simple strategy for demonstration"""
    def __init__(self, name):
        self.name = name

    def generate_signal(self, data):
        # Always return 0 (hold) for demo
        return 0

async def run_demo():
    """Run a simple demo of the real-time trading system"""

    print("=" * 70)
    print("REAL-TIME TRADING SYSTEM DEMO")
    print("=" * 70)
    print()

    # Check for API keys
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print("⚠️  API keys not found!")
        print()
        print("Please set environment variables:")
        print("  export ALPACA_API_KEY='your_key_here'")
        print("  export ALPACA_API_SECRET='your_secret_here'")
        print()
        print("Or edit this file and add your keys directly.")
        print()
        print("Get free API keys at: https://alpaca.markets")
        print("=" * 70)
        return

    print(f"✓ API keys found")
    print(f"✓ Using paper trading (no real money)")
    print()

    # Initialize connector
    print("Connecting to Alpaca Markets...")
    connector = AlpacaConnector(
        api_key=api_key,
        api_secret=api_secret,
        paper=True  # Paper trading
    )

    # Register tick handler
    tick_count = [0]  # Use list for mutable counter

    @connector.on_tick
    async def handle_tick(tick: MarketTick):
        tick_count[0] += 1

        # Print every 10th tick
        if tick_count[0] % 10 == 0:
            print(f"[{tick.timestamp.strftime('%H:%M:%S')}] "
                  f"{tick.symbol}: ${tick.price:.2f} "
                  f"(Bid: ${tick.bid:.2f}, Ask: ${tick.ask:.2f}) "
                  f"[{tick_count[0]} ticks]")

    try:
        # Connect
        await connector.start()
        print("✓ Connected to Alpaca WebSocket")
        print()

        # Subscribe to symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        await connector.subscribe(symbols)
        print(f"✓ Subscribed to: {', '.join(symbols)}")
        print()
        print("Receiving real-time market data...")
        print("(Press Ctrl+C to stop)")
        print("-" * 70)

        # Run for 60 seconds or until interrupted
        await asyncio.sleep(60)

    except KeyboardInterrupt:
        print()
        print("-" * 70)
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await connector.stop()
        print()
        print("=" * 70)
        print(f"DEMO COMPLETE")
        print(f"Total ticks received: {tick_count[0]}")
        print("=" * 70)


async def run_full_trading_demo():
    """
    Full trading demo with strategy router and P&L tracking
    (Currently simplified - strategies need proper implementation)
    """

    print("=" * 70)
    print("FULL TRADING SYSTEM DEMO")
    print("=" * 70)
    print()

    # Get API keys
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return

    # Initialize components
    connector = AlpacaConnector(api_key=api_key, api_secret=api_secret, paper=True)

    # Create simple strategies
    strategies = {
        'momentum': SimpleStrategy('momentum'),
        'mean_reversion': SimpleStrategy('mean_reversion'),
        'market_making': SimpleStrategy('market_making'),
    }

    # Initialize strategy router
    router = AdaptiveStrategyRouter(strategies)

    # Initialize P&L tracker
    pnl_tracker = RealTimePnLTracker(initial_capital=100000.0)

    print("✓ Strategy Router initialized")
    print("✓ P&L Tracker initialized")
    print("✓ Starting trading engine...")
    print()

    # Data collection
    market_data = {}

    @connector.on_tick
    async def handle_tick(tick: MarketTick):
        # Update market data
        market_data[tick.symbol] = {
            'price': tick.price,
            'volume': tick.volume,
            'bid': tick.bid,
            'ask': tick.ask,
        }

        # Update P&L tracker
        pnl_tracker.update_market_price(tick.symbol, tick.price)

        # Select strategy (every 10 ticks)
        if len(market_data) > 0:
            strategy_name, confidence = router.select_strategy(market_data[tick.symbol])

            # Print status
            regime = router.get_regime_info()
            print(f"[{tick.timestamp.strftime('%H:%M:%S')}] "
                  f"{tick.symbol}: ${tick.price:.2f} | "
                  f"Strategy: {strategy_name} ({confidence:.2f}) | "
                  f"Regime: {regime.get('regime', 'unknown')} | "
                  f"P&L: ${pnl_tracker.get_total_pnl():.2f}")

    try:
        await connector.start()
        await connector.subscribe(['AAPL', 'MSFT', 'GOOGL'])

        print("Trading system running...")
        print("(Press Ctrl+C to stop)")
        print("-" * 70)

        await asyncio.sleep(60)

    except KeyboardInterrupt:
        print()
        print("-" * 70)
    finally:
        await connector.stop()

        # Print final summary
        print()
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        pnl_tracker.print_summary()

        print("\nStrategy Performance:")
        print(router.get_performance_summary().to_string(index=False))


if __name__ == "__main__":
    print()
    print("Choose demo mode:")
    print("1. Simple Demo (just receive market data)")
    print("2. Full Trading Demo (with strategies and P&L)")
    print()

    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        asyncio.run(run_demo())
    elif choice == "2":
        asyncio.run(run_full_trading_demo())
    else:
        print("Invalid choice")

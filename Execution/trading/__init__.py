"""
Real-Time Trading System

Complete real-time trading system with:
- Market data connectors (Alpaca, Binance, etc.)
- Adaptive strategy selection
- Real-time P&L tracking
- Automated execution

Usage:
    from realtime_trading import RealTimeTradingEngine
    from realtime_trading.connectors import AlpacaConnector

    engine = RealTimeTradingEngine(
        connector=AlpacaConnector(...),
        strategies={...},
        initial_capital=100000
    )

    await engine.start()
"""

from .trading_engine import RealTimeTradingEngine
from Market_Data.base_connector import BaseConnector, MarketTick, OrderBookSnapshot
from Market_Data.alpaca_connector import AlpacaConnector
from .strategy_router.adaptive_router import AdaptiveStrategyRouter, MarketRegime
from Monitoring.realtime_pnl import RealTimePnLTracker, Position, Trade

__all__ = [
    'RealTimeTradingEngine',
    'BaseConnector',
    'MarketTick',
    'OrderBookSnapshot',
    'AlpacaConnector',
    'AdaptiveStrategyRouter',
    'MarketRegime',
    'RealTimePnLTracker',
    'Position',
    'Trade',
]

__version__ = '1.0.0'

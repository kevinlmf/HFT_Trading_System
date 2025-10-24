"""
Base API Connector for Real-Time Market Data

This module provides the abstract base class for all market data connectors.
Supports multiple data sources: WebSocket, REST API, FIX protocol, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
        }


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[tuple]  # [(price, size), ...]
    asks: List[tuple]  # [(price, size), ...]

    def get_spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0

    def get_mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2.0
        return 0.0


class BaseConnector(ABC):
    """
    Abstract base class for market data connectors

    Usage:
        connector = YourConnector(api_key="...")
        connector.subscribe(['AAPL', 'MSFT'])

        @connector.on_tick
        def handle_tick(tick: MarketTick):
            print(f"Received: {tick.symbol} @ {tick.price}")

        await connector.start()
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_connected = False
        self.subscribed_symbols: List[str] = []

        # Callbacks
        self._tick_callbacks: List[Callable] = []
        self._orderbook_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []

        self._connection_task: Optional[asyncio.Task] = None

    @abstractmethod
    async def connect(self):
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection"""
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        pass

    def on_tick(self, callback: Callable[[MarketTick], None]):
        """Register callback for market ticks"""
        self._tick_callbacks.append(callback)
        return callback

    def on_orderbook(self, callback: Callable[[OrderBookSnapshot], None]):
        """Register callback for order book updates"""
        self._orderbook_callbacks.append(callback)
        return callback

    def on_error(self, callback: Callable[[Exception], None]):
        """Register callback for errors"""
        self._error_callbacks.append(callback)
        return callback

    async def _emit_tick(self, tick: MarketTick):
        """Emit tick to all registered callbacks"""
        for callback in self._tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick)
                else:
                    callback(tick)
            except Exception as e:
                logger.error(f"Error in tick callback: {e}")
                await self._emit_error(e)

    async def _emit_orderbook(self, orderbook: OrderBookSnapshot):
        """Emit order book to all registered callbacks"""
        for callback in self._orderbook_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(orderbook)
                else:
                    callback(orderbook)
            except Exception as e:
                logger.error(f"Error in orderbook callback: {e}")
                await self._emit_error(e)

    async def _emit_error(self, error: Exception):
        """Emit error to all registered callbacks"""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def start(self):
        """Start the connector"""
        if not self.is_connected:
            await self.connect()
            logger.info(f"{self.__class__.__name__} started")

    async def stop(self):
        """Stop the connector"""
        if self.is_connected:
            await self.disconnect()
            logger.info(f"{self.__class__.__name__} stopped")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"connected={self.is_connected}, "
                f"symbols={len(self.subscribed_symbols)})")

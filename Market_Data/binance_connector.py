"""
Binance Cryptocurrency Connector

支持加密货币现货和期货的实时数据
WebSocket实时数据流
"""

import asyncio
import json
from typing import List
from datetime import datetime
import logging

try:
    from binance import AsyncClient, BinanceSocketManager
except ImportError:
    AsyncClient = None
    BinanceSocketManager = None

from .base_connector import BaseConnector, MarketTick

logger = logging.getLogger(__name__)


class BinanceConnector(BaseConnector):
    """
    Binance cryptocurrency connector
    
    支持实时加密货币数据（BTC, ETH等）
    
    Usage:
        connector = BinanceConnector(
            api_key="your_key",  # 可选，仅数据不需要
            api_secret="your_secret"  # 可选
        )
        await connector.subscribe(['BTCUSDT', 'ETHUSDT'])
        
        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")
        
        await connector.start()
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Args:
            api_key: Binance API密钥（数据订阅不需要）
            api_secret: Binance API密钥（数据订阅不需要）
            testnet: 是否使用测试网
        """
        super().__init__(api_key, api_secret)
        self.testnet = testnet
        self.client = None
        self.socket_manager = None
        self._sockets = {}
    
    async def connect(self):
        """建立连接"""
        if AsyncClient is None:
            raise ImportError("Please install python-binance: pip install python-binance")
        
        try:
            # Try to create client - this will ping the API
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            self.socket_manager = BinanceSocketManager(self.client)
            self.is_connected = True
            
            logger.info(f"Binance connector connected (testnet={self.testnet})")
        except Exception as e:
            error_msg = str(e)
            if "restricted location" in error_msg.lower() or "eligibility" in error_msg.lower():
                logger.error("=" * 80)
                logger.error("⚠️  Binance API is not available in your region")
                logger.error("=" * 80)
                logger.error("")
                logger.error("Binance restricts access from certain regions (e.g., USA)")
                logger.error("")
                logger.error("Suggested alternatives:")
                logger.error("  1. Use Yahoo Finance (free, no API key):")
                logger.error("     ./run_trading.sh paper --connector yahoo --symbols AAPL --interval 10")
                logger.error("")
                logger.error("  2. Use Alpaca Markets (free paper trading):")
                logger.error("     export ALPACA_API_KEY='your_key'")
                logger.error("     export ALPACA_API_SECRET='your_secret'")
                logger.error("     ./run_trading.sh paper --connector alpaca --symbols AAPL --interval 5")
                logger.error("")
                logger.error("  3. Use Coinbase Pro (if available in your region):")
                logger.error("     ./run_trading.sh paper --connector coinbase --symbols BTC-USD --interval 10")
                logger.error("")
                logger.error("=" * 80)
                raise ConnectionError(
                    "Binance API is not available in your region. "
                    "Please use another connector (yahoo, alpaca, or coinbase)."
                ) from e
            else:
                logger.error(f"Failed to connect to Binance: {e}")
                raise
    
    async def disconnect(self):
        """断开连接"""
        # 关闭所有socket
        for socket in self._sockets.values():
            await socket.__aexit__(None, None, None)
        self._sockets.clear()
        
        if self.client:
            await self.client.close_connection()
        
        self.is_connected = False
        logger.info("Binance connector disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的（Binance使用ticker格式，如BTCUSDT）"""
        self.subscribed_symbols = symbols
        
        if not self.is_connected:
            await self.connect()
        
        # 为每个symbol创建WebSocket连接
        for symbol in symbols:
            await self._subscribe_symbol(symbol)
        
        logger.info(f"Subscribed to {len(symbols)} symbols on Binance")
    
    async def _subscribe_symbol(self, symbol: str):
        """订阅单个标的"""
        try:
            # 使用ticker socket获取实时价格更新
            socket = self.socket_manager.ticker_socket(symbol.lower())
            
            async with socket as ts:
                self._sockets[symbol] = ts
                
                async for msg in ts:
                    if not self.is_connected:
                        break
                    
                    data = json.loads(msg)
                    await self._process_binance_message(symbol, data)
        
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
    
    async def _process_binance_message(self, symbol: str, data: dict):
        """处理Binance消息并转换为MarketTick"""
        try:
            # Binance ticker消息格式
            price = float(data.get('c', 0))  # 最新价格
            volume = float(data.get('v', 0))  # 24小时成交量
            bid = float(data.get('b', price))  # 最佳买价
            ask = float(data.get('a', price))  # 最佳卖价
            
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
            )
            
            await self._emit_tick(tick)
        
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self._sockets:
                await self._sockets[symbol].__aexit__(None, None, None)
                del self._sockets[symbol]
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols")







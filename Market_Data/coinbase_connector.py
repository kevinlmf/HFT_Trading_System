"""
Coinbase Pro Connector

加密货币交易所API
支持实时WebSocket数据
"""

import asyncio
import json
from typing import List
from datetime import datetime
import logging

try:
    import websockets
except ImportError:
    websockets = None

from .base_connector import BaseConnector, MarketTick

logger = logging.getLogger(__name__)


class CoinbaseProConnector(BaseConnector):
    """
    Coinbase Pro (Advanced Trade) connector
    
    支持实时加密货币数据
    
    Usage:
        connector = CoinbaseProConnector()
        await connector.subscribe(['BTC-USD', 'ETH-USD'])
        
        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")
        
        await connector.start()
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, sandbox: bool = True):
        """
        Args:
            api_key: Coinbase Pro API密钥（数据订阅不需要）
            api_secret: Coinbase Pro API密钥（数据订阅不需要）
            sandbox: 是否使用沙盒环境
        """
        super().__init__(api_key, api_secret)
        self.sandbox = sandbox
        # Coinbase Advanced Trade WebSocket URL
        # Note: Coinbase Pro legacy API may be deprecated
        self.ws_url = (
            "wss://advanced-trade-ws.coinbase.com" if not sandbox
            else "wss://advanced-trade-ws.coinbase.com"  # Sandbox uses same endpoint
        )
        # Fallback to legacy Coinbase Pro WebSocket
        self.legacy_ws_url = (
            "wss://ws-feed.pro.coinbase.com" if not sandbox
            else "wss://ws-feed-public.sandbox.pro.coinbase.com"
        )
        self.ws = None
        self.use_legacy = False
        self._message_loop_task = None
        self._msg_count = 0
    
    async def connect(self):
        """建立WebSocket连接"""
        if websockets is None:
            raise ImportError("Please install websockets: pip install websockets")
        
        if self.is_connected:
            logger.info("Coinbase Pro connector already connected")
            return
        
        # Try legacy Coinbase Pro WebSocket first (more stable)
        # Advanced Trade API may require authentication
        urls_to_try = [
            (self.legacy_ws_url, True),  # Try legacy first
            (self.ws_url, False)  # Fallback to Advanced Trade
        ]
        
        last_error = None
        for url, is_legacy in urls_to_try:
            try:
                logger.info(f"Attempting to connect to Coinbase {'Legacy' if is_legacy else 'Advanced Trade'} WebSocket: {url}")
                self.ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
                self.is_connected = True
                self.use_legacy = is_legacy
                logger.info(f"✓ Coinbase Pro connector connected (sandbox={self.sandbox}, legacy={is_legacy})")
                
                # 启动消息接收循环（如果还没有启动）
                if self._message_loop_task is None or self._message_loop_task.done():
                    self._message_loop_task = asyncio.create_task(self._message_loop())
                    logger.info("Message loop task started")
                
                # Wait a bit for connection to stabilize
                await asyncio.sleep(0.5)
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to connect to {url}: {e}")
                continue
        
        # If all URLs failed, raise the last error
        logger.error(f"Failed to connect to Coinbase Pro WebSocket after trying all endpoints: {last_error}")
        raise ConnectionError(f"Unable to connect to Coinbase Pro WebSocket: {last_error}")
    
    async def disconnect(self):
        """断开连接"""
        self.is_connected = False
        
        if self._message_loop_task and not self._message_loop_task.done():
            self._message_loop_task.cancel()
            try:
                await self._message_loop_task
            except asyncio.CancelledError:
                pass
        
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            self.ws = None
        
        logger.info("Coinbase Pro connector disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的（Coinbase Pro使用格式如BTC-USD）"""
        if not self.is_connected or self.ws is None:
            await self.connect()
        
        # Wait a moment for connection to be fully established
        await asyncio.sleep(0.5)
        
        # Ensure WebSocket is still valid
        if self.ws is None or not self.is_connected:
            raise ConnectionError("WebSocket connection not available for subscription")
        
        self.subscribed_symbols = symbols
        
        # Coinbase Pro订阅消息格式（legacy API）
        if self.use_legacy:
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": symbols,
                "channels": ["ticker"]  # ticker频道提供最新价格
            }
        else:
            # Advanced Trade API format (may need different format)
            # For now, try legacy format as fallback
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": symbols,
                "channels": ["ticker"]
            }
        
        logger.info(f"📤 Sending subscription message for {len(symbols)} symbols: {symbols}")
        try:
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"✓ Subscribed to {len(symbols)} symbols on Coinbase Pro (legacy={self.use_legacy})")
        except Exception as e:
            logger.error(f"Failed to send subscription message: {e}")
            raise
        
        # Wait a moment for subscription to be processed
        await asyncio.sleep(0.5)
    
    async def _message_loop(self):
        """接收WebSocket消息"""
        reconnect_attempts = 0
        max_reconnect_attempts = 3
        
        while True:
            try:
                if not self.is_connected or self.ws is None:
                    logger.warning("WebSocket not connected, attempting to reconnect...")
                    await self.connect()
                    
                    # Resubscribe after reconnection
                    if hasattr(self, 'subscribed_symbols') and self.subscribed_symbols:
                        logger.info(f"Resubscribing to {len(self.subscribed_symbols)} symbols after reconnection...")
                        await self.subscribe(self.subscribed_symbols)
                    
                    reconnect_attempts = 0
                
                # Receive messages
                while self.is_connected and self.ws:
                    try:
                        message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                        data = json.loads(message)
                        await self._process_message(data)
                        reconnect_attempts = 0  # Reset on successful message
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        if self.ws and self.is_connected:
                            try:
                                await self.ws.ping()
                                logger.debug("Sent ping to Coinbase WebSocket")
                            except Exception as e:
                                logger.warning(f"Failed to send ping: {e}")
                                break
            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Coinbase Pro WebSocket connection closed: {e}")
                self.is_connected = False
                
                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    wait_time = min(2 ** reconnect_attempts, 10)  # Exponential backoff
                    logger.info(f"Attempting to reconnect ({reconnect_attempts}/{max_reconnect_attempts}) in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max reconnection attempts reached. Stopping message loop.")
                    break
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse WebSocket message: {e}")
                logger.debug(f"Raw message: {message if 'message' in locals() else 'N/A'}")
                continue
                
            except Exception as e:
                logger.error(f"Error in Coinbase Pro message loop: {e}", exc_info=True)
                self.is_connected = False
                
                if reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    wait_time = min(2 ** reconnect_attempts, 10)
                    logger.info(f"Attempting to reconnect after error ({reconnect_attempts}/{max_reconnect_attempts}) in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max reconnection attempts reached. Stopping message loop.")
                    break
    
    async def _process_message(self, data: dict):
        """处理Coinbase Pro消息并转换为MarketTick"""
        try:
            msg_type = data.get('type')
            
            # Log all message types for debugging (first few messages)
            if not hasattr(self, '_msg_count'):
                self._msg_count = 0
            self._msg_count += 1
            
            if self._msg_count <= 5 or msg_type not in ['subscriptions', 'heartbeat']:
                logger.info(f"📨 Coinbase message #{self._msg_count}: type={msg_type}, keys={list(data.keys())[:5]}")
            
            if msg_type == 'ticker':
                symbol = data.get('product_id')
                price = float(data.get('price', 0))
                volume_24h = float(data.get('volume_24h', 0))
                bid = float(data.get('best_bid', price))
                ask = float(data.get('best_ask', price))
                
                if price > 0 and symbol:
                    tick = MarketTick(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        price=price,
                        volume=volume_24h,
                        bid=bid,
                        ask=ask,
                    )
                    
                    logger.info(f"📈 Coinbase tick: {symbol} = ${price:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f})")
                    await self._emit_tick(tick)
                else:
                    logger.warning(f"Invalid ticker data: symbol={symbol}, price={price}, data={data}")
            
            elif msg_type == 'subscriptions':
                logger.info(f"✓ Coinbase subscription confirmed: {data}")
                # Resubscribe if needed
                if hasattr(self, 'subscribed_symbols') and self.subscribed_symbols:
                    logger.info(f"Subscription active for: {self.subscribed_symbols}")
            
            elif msg_type == 'heartbeat':
                # Heartbeat messages keep connection alive
                if self._msg_count % 100 == 0:  # Log every 100th heartbeat
                    logger.debug(f"Heartbeat received (message #{self._msg_count})")
            
            elif msg_type == 'error':
                error_msg = data.get('message', 'Unknown error')
                logger.error(f"❌ Coinbase Pro error: {error_msg}")
                logger.error(f"Full error data: {data}")
                # Don't disconnect on error, let reconnection handle it
            
            elif msg_type == 'snapshot':
                # Level2 order book snapshot - not needed for ticker
                logger.debug(f"Received snapshot for {data.get('product_id', 'unknown')}")
            
            elif msg_type == 'l2update':
                # Level2 order book update - not needed for ticker
                logger.debug(f"Received l2update for {data.get('product_ids', ['unknown'])}")
            
            else:
                logger.debug(f"Unhandled message type: {msg_type}, data keys: {list(data.keys())}")
        
        except Exception as e:
            logger.error(f"Error processing Coinbase Pro message: {e}", exc_info=True)
            logger.error(f"Problematic data: {data if 'data' in locals() else 'N/A'}")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        unsubscribe_msg = {
            "type": "unsubscribe",
            "product_ids": symbols,
            "channels": ["ticker"]
        }
        
        await self.ws.send(json.dumps(unsubscribe_msg))
        
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols")


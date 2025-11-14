"""
Polygon.io Stock Market Connector

专业的股票市场数据API
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


class PolygonConnector(BaseConnector):
    """
    Polygon.io connector for stock market data
    
    支持实时股票报价和历史数据
    
    Usage:
        connector = PolygonConnector(api_key="your_polygon_key")
        await connector.subscribe(['AAPL', 'MSFT'])
        
        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")
        
        await connector.start()
    """
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: Polygon.io API密钥
        """
        super().__init__(api_key)
        self.ws_url = "wss://socket.polygon.io/stocks"
        self.ws = None
        self._authenticated = False
    
    async def connect(self):
        """建立WebSocket连接"""
        if websockets is None:
            raise ImportError("Please install websockets: pip install websockets")
        
        try:
            self.ws = await websockets.connect(self.ws_url)
            
            # 发送认证消息
            auth_msg = {
                "action": "auth",
                "params": self.api_key
            }
            await self.ws.send(json.dumps(auth_msg))
            
            # 等待认证响应
            response = await self.ws.recv()
            auth_response = json.loads(response)
            
            if auth_response[0].get("status") == "auth_success":
                self._authenticated = True
                self.is_connected = True
                logger.info("Polygon.io authenticated successfully")
            else:
                raise Exception("Polygon.io authentication failed")
        
        except Exception as e:
            logger.error(f"Failed to connect to Polygon.io: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.ws:
            await self.ws.close()
        self.is_connected = False
        self._authenticated = False
        logger.info("Polygon.io disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的"""
        if not self.is_connected or not self._authenticated:
            await self.connect()
        
        self.subscribed_symbols = symbols
        
        # 订阅实时报价
        subscribe_msg = {
            "action": "subscribe",
            "params": ",".join([f"Q.{s}" for s in symbols])  # Q = Quote
        }
        
        await self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(symbols)} symbols on Polygon.io")
        
        # 启动消息接收循环
        asyncio.create_task(self._message_loop())
    
    async def _message_loop(self):
        """接收WebSocket消息"""
        try:
            while self.is_connected:
                message = await self.ws.recv()
                data = json.loads(message)
                
                # 处理多个事件
                for event in data:
                    await self._process_polygon_message(event)
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Polygon.io WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in Polygon.io message loop: {e}")
    
    async def _process_polygon_message(self, event: dict):
        """处理Polygon消息并转换为MarketTick"""
        try:
            event_type = event.get("ev")
            
            if event_type == "Q":  # Quote事件
                symbol = event.get("sym")
                bid = event.get("bp", 0)  # Bid price
                ask = event.get("ap", 0)  # Ask price
                bid_size = event.get("bs", 0)  # Bid size
                ask_size = event.get("as", 0)  # Ask size
                
                # 使用中间价作为价格
                price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid or ask
                
                tick = MarketTick(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(event.get("t", 0) / 1000),
                    price=price,
                    volume=0,  # Quote事件不包含成交量
                    bid=bid,
                    ask=ask,
                    bid_size=bid_size,
                    ask_size=ask_size,
                )
                
                await self._emit_tick(tick)
            
            elif event_type == "T":  # Trade事件
                symbol = event.get("sym")
                price = event.get("p", 0)
                volume = event.get("s", 0)
                
                tick = MarketTick(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(event.get("t", 0) / 1000),
                    price=price,
                    volume=volume,
                )
                
                await self._emit_tick(tick)
        
        except Exception as e:
            logger.error(f"Error processing Polygon message: {e}")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        unsubscribe_msg = {
            "action": "unsubscribe",
            "params": ",".join([f"Q.{s}" for s in symbols])
        }
        
        await self.ws.send(json.dumps(unsubscribe_msg))
        
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols")













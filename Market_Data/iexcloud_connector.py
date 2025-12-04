"""
IEX Cloud Connector

股票市场数据API
免费层级：50,000 messages/month
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


class IEXCloudConnector(BaseConnector):
    """
    IEX Cloud connector for stock market data
    
    免费层级：50,000 messages/month
    
    Usage:
        connector = IEXCloudConnector(api_key="your_key")
        await connector.subscribe(['AAPL', 'MSFT'])
        
        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")
        
        await connector.start()
    """
    
    def __init__(self, api_key: str, sandbox: bool = True):
        """
        Args:
            api_key: IEX Cloud API密钥
            sandbox: 是否使用沙盒环境
        """
        super().__init__(api_key)
        self.sandbox = sandbox
        self.ws_url = (
            "wss://ws-api.iextrading.com/1.0/tops" if not sandbox
            else "wss://sandbox-sse.iexapis.com/stable/tops"
        )
        self.ws = None
    
    async def connect(self):
        """建立WebSocket连接"""
        if websockets is None:
            raise ImportError("Please install websockets: pip install websockets")
        
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.is_connected = True
            logger.info(f"IEX Cloud connector connected (sandbox={self.sandbox})")
            
            # 启动消息接收循环
            asyncio.create_task(self._message_loop())
        except Exception as e:
            logger.error(f"Failed to connect to IEX Cloud: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.ws:
            await self.ws.close()
        self.is_connected = False
        logger.info("IEX Cloud connector disconnected")
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的"""
        if not self.is_connected:
            await self.connect()
        
        self.subscribed_symbols = symbols
        
        # IEX Cloud订阅消息格式
        subscribe_msg = json.dumps(symbols)
        
        await self.ws.send(subscribe_msg)
        logger.info(f"Subscribed to {len(symbols)} symbols on IEX Cloud")
    
    async def _message_loop(self):
        """接收WebSocket消息"""
        try:
            while self.is_connected:
                message = await self.ws.recv()
                data = json.loads(message)
                
                await self._process_message(data)
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("IEX Cloud WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in IEX Cloud message loop: {e}")
    
    async def _process_message(self, data: dict):
        """处理IEX Cloud消息并转换为MarketTick"""
        try:
            symbol = data.get('symbol')
            price = float(data.get('lastSalePrice', 0))
            size = float(data.get('lastSaleSize', 0))
            bid = float(data.get('bidPrice', price))
            ask = float(data.get('askPrice', price))
            bid_size = float(data.get('bidSize', 0))
            ask_size = float(data.get('askSize', 0))
            
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=price,
                volume=size,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
            )
            
            await self._emit_tick(tick)
        
        except Exception as e:
            logger.error(f"Error processing IEX Cloud message: {e}")
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        # IEX Cloud使用空数组取消订阅
        unsubscribe_msg = json.dumps([])
        await self.ws.send(unsubscribe_msg)
        
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols")








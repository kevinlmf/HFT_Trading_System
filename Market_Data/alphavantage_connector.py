"""
Alpha Vantage Connector

免费股票市场数据API
500 calls/day免费额度
"""

import asyncio
import requests
from typing import List
from datetime import datetime
import logging
import time

from .base_connector import BaseConnector, MarketTick

logger = logging.getLogger(__name__)


class AlphaVantageConnector(BaseConnector):
    """
    Alpha Vantage connector for stock market data
    
    免费API，500 calls/day
    
    Usage:
        connector = AlphaVantageConnector(api_key="your_key")
        await connector.subscribe(['AAPL', 'MSFT'])
        
        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")
        
        await connector.start()
    """
    
    def __init__(self, api_key: str, update_interval: float = 60.0):
        """
        Args:
            api_key: Alpha Vantage API密钥（免费注册：https://www.alphavantage.co/support/#api-key）
            update_interval: 更新间隔（秒），默认60秒（避免超过免费限制）
        """
        super().__init__(api_key)
        self.update_interval = update_interval
        self.base_url = "https://www.alphavantage.co/query"
        self._update_task = None
        self._call_count = 0
        self._max_calls_per_day = 500
    
    async def connect(self):
        """建立连接"""
        self.is_connected = True
        logger.info("Alpha Vantage connector started (500 calls/day free limit)")
    
    async def disconnect(self):
        """断开连接"""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        self.is_connected = False
        logger.info(f"Alpha Vantage connector stopped (used {self._call_count} API calls)")
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的"""
        self.subscribed_symbols = symbols
        
        # 确保已连接
        if not self.is_connected:
            await self.connect()
        
        logger.info(f"Subscribed to {len(symbols)} symbols via Alpha Vantage")
        
        # 启动更新任务
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._update_loop())
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols")
    
    async def _update_loop(self):
        """定期更新数据"""
        update_count = 0
        while self.is_connected:
            try:
                for symbol in list(self.subscribed_symbols):
                    # 检查API调用限制
                    if self._call_count >= self._max_calls_per_day:
                        logger.warning(f"⚠️  Alpha Vantage daily limit reached ({self._max_calls_per_day} calls)")
                        await asyncio.sleep(3600)  # 等待1小时
                        self._call_count = 0  # 重置计数器
                    
                    await self._fetch_tick(symbol)
                    await asyncio.sleep(1)  # 避免请求过快
                
                update_count += 1
                if update_count % 10 == 0:
                    logger.info(f"Alpha Vantage: Updated {update_count} times, "
                               f"API calls used: {self._call_count}/{self._max_calls_per_day}")
                
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Alpha Vantage update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _fetch_tick(self, symbol: str):
        """获取单个标的的tick数据"""
        try:
            # 使用TIME_SERIES_INTRADAY获取最新价格
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': '1min',
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            self._call_count += 1
            
            if response.status_code != 200:
                logger.error(f"Alpha Vantage API error for {symbol}: {response.status_code}")
                return
            
            data = response.json()
            
            # 检查错误
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return
            
            # 解析数据
            if 'Time Series (1min)' not in data:
                logger.warning(f"No time series data for {symbol}")
                return
            
            time_series = data['Time Series (1min)']
            if not time_series:
                return
            
            # 获取最新数据点
            latest_time = max(time_series.keys())
            latest_data = time_series[latest_time]
            
            price = float(latest_data['4. close'])
            volume = float(latest_data['5. volume'])
            
            # 创建MarketTick
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                bid=float(latest_data['3. low']),  # 使用low作为bid
                ask=float(latest_data['2. high']),  # 使用high作为ask
            )
            
            await self._emit_tick(tick)
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage tick for {symbol}: {e}", exc_info=True)








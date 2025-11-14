"""
Yahoo Finance Connector (yfinance)

免费股票数据源，无需API密钥
适合测试和开发使用
"""

import asyncio
import yfinance as yf
from typing import List
from datetime import datetime
import logging
import pandas as pd

from .base_connector import BaseConnector, MarketTick

logger = logging.getLogger(__name__)


class YahooFinanceConnector(BaseConnector):
    """
    Yahoo Finance connector using yfinance library
    
    注意: Yahoo Finance数据有15-20分钟延迟，不适合实时交易
    适合用于回测和策略开发
    
    Usage:
        connector = YahooFinanceConnector()
        await connector.subscribe(['AAPL', 'MSFT'])
        
        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")
        
        await connector.start()
    """
    
    def __init__(self, update_interval: float = 5.0):
        """
        Args:
            update_interval: 更新间隔（秒），默认5秒
        """
        super().__init__()
        self.update_interval = update_interval
        self._tickers = {}
        self._update_task = None
    
    async def connect(self):
        """建立连接（Yahoo Finance不需要实际连接）"""
        self.is_connected = True
        logger.info("Yahoo Finance connector started (note: data has 15-20min delay)")
    
    async def disconnect(self):
        """断开连接"""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        self.is_connected = False
        logger.info("Yahoo Finance connector stopped")
    
    async def subscribe(self, symbols: List[str]):
        """订阅标的"""
        self.subscribed_symbols = symbols
        
        # 确保已连接
        if not self.is_connected:
            await self.connect()
        
        # 创建yfinance ticker对象
        for symbol in symbols:
            self._tickers[symbol] = yf.Ticker(symbol)
        
        logger.info(f"Subscribed to {len(symbols)} symbols via Yahoo Finance")
        
        # 启动更新任务（如果还没有启动）
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._update_loop())
    
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self._tickers:
                del self._tickers[symbol]
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols")
    
    async def _update_loop(self):
        """定期更新数据"""
        update_count = 0
        while self.is_connected:
            try:
                for symbol in list(self._tickers.keys()):
                    await self._fetch_tick(symbol)
                
                update_count += 1
                if update_count % 6 == 0:  # Log every 6 updates (about 1 minute with 10s interval)
                    logger.info(f"Yahoo Finance: Updated {update_count} times, {len(self._tickers)} symbols")
                
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _fetch_tick(self, symbol: str):
        """获取单个标的的tick数据"""
        try:
            ticker = self._tickers[symbol]
            
            # 获取最新价格（使用更短的时间窗口以提高更新频率）
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            current_volume = float(latest['Volume'])
            
            # 获取bid/ask（如果可用）
            try:
                info = ticker.info
                bid = float(info.get('bid', current_price))
                ask = float(info.get('ask', current_price))
            except:
                # 如果info不可用，使用价格作为bid/ask
                bid = current_price - 0.01
                ask = current_price + 0.01
            
            # 创建MarketTick
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=current_volume,
                bid=bid,
                ask=ask,
            )
            
            logger.debug(f"Yahoo Finance tick for {symbol}: ${current_price:.2f}, volume={current_volume:.0f}")
            await self._emit_tick(tick)
            
        except Exception as e:
            logger.error(f"Error fetching tick for {symbol}: {e}", exc_info=True)



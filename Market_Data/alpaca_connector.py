"""
Alpaca API Connector for Real-Time Stock Data

Free tier available at https://alpaca.markets
Supports real-time stock quotes via WebSocket
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


class AlpacaConnector(BaseConnector):
    """
    Alpaca Markets WebSocket connector for real-time stock data

    Usage:
        connector = AlpacaConnector(
            api_key="YOUR_API_KEY",
            api_secret="YOUR_API_SECRET"
        )
        connector.subscribe(['AAPL', 'MSFT', 'GOOGL'])

        @connector.on_tick
        async def handle_tick(tick):
            print(f"{tick.symbol}: ${tick.price}")

        await connector.start()
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        super().__init__(api_key, api_secret)
        self.paper = paper
        self.ws_url = (
            "wss://stream.data.alpaca.markets/v2/iex"  # Free IEX data
            if not paper else
            "wss://stream.data.alpaca.markets/v2/test"  # Paper trading
        )
        self.ws = None

    async def connect(self):
        """Establish WebSocket connection"""
        if websockets is None:
            raise ImportError("websockets library not installed. Run: pip install websockets")

        try:
            self.ws = await websockets.connect(self.ws_url)
            self.is_connected = True

            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await self.ws.send(json.dumps(auth_msg))

            # Wait for auth response
            response = await self.ws.recv()
            auth_response = json.loads(response)

            if auth_response[0].get('T') == 'success':
                logger.info("Alpaca WebSocket authenticated successfully")
            else:
                raise ConnectionError(f"Authentication failed: {auth_response}")

            # Start message handler
            asyncio.create_task(self._message_handler())

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("Alpaca WebSocket disconnected")

    async def subscribe(self, symbols: List[str]):
        """Subscribe to real-time quotes"""
        if not self.is_connected:
            raise ConnectionError("Not connected. Call connect() first.")

        self.subscribed_symbols.extend(symbols)

        subscribe_msg = {
            "action": "subscribe",
            "quotes": symbols,  # Real-time quotes
            "trades": symbols,  # Real-time trades
        }

        await self.ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(symbols)} symbols: {symbols}")

    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if not self.is_connected:
            return

        unsubscribe_msg = {
            "action": "unsubscribe",
            "quotes": symbols,
            "trades": symbols,
        }

        await self.ws.send(json.dumps(unsubscribe_msg))

        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)

        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.ws:
                data = json.loads(message)

                for item in data:
                    msg_type = item.get('T')

                    # Trade message
                    if msg_type == 't':
                        tick = MarketTick(
                            symbol=item['S'],
                            timestamp=datetime.fromisoformat(item['t'].replace('Z', '+00:00')),
                            price=float(item['p']),
                            volume=float(item['s']),
                        )
                        await self._emit_tick(tick)

                    # Quote message (bid/ask)
                    elif msg_type == 'q':
                        tick = MarketTick(
                            symbol=item['S'],
                            timestamp=datetime.fromisoformat(item['t'].replace('Z', '+00:00')),
                            price=(float(item['bp']) + float(item['ap'])) / 2,  # Mid price
                            volume=0,
                            bid=float(item['bp']),
                            ask=float(item['ap']),
                            bid_size=float(item['bs']),
                            ask_size=float(item['as']),
                        )
                        await self._emit_tick(tick)

                    # Error message
                    elif msg_type == 'error':
                        logger.error(f"Alpaca error: {item}")
                        await self._emit_error(Exception(item.get('msg', 'Unknown error')))

        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            await self._emit_error(e)
            self.is_connected = False


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize connector
        connector = AlpacaConnector(
            api_key="YOUR_API_KEY",
            api_secret="YOUR_API_SECRET",
            paper=True
        )

        # Register tick handler
        @connector.on_tick
        async def handle_tick(tick: MarketTick):
            print(f"[{tick.timestamp}] {tick.symbol}: ${tick.price:.2f} "
                  f"(Bid: ${tick.bid:.2f}, Ask: ${tick.ask:.2f})")

        # Start connector
        await connector.start()
        await connector.subscribe(['AAPL', 'MSFT', 'GOOGL'])

        # Keep running
        await asyncio.sleep(60)  # Run for 60 seconds

        # Cleanup
        await connector.stop()

    asyncio.run(main())

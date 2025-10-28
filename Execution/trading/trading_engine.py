"""
Real-Time Trading Engine

Main trading loop that:
1. Receives market data from API
2. Detects market regime
3. Selects best strategy
4. Generates signals
5. Executes trades
6. Tracks P&L
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import signal
import sys

# Import our modules
from Data.connectors.base_connector import MarketTick
from Execution.trading.strategy_router.adaptive_router import AdaptiveStrategyRouter
from Monitoring.pnl_tracking.realtime_pnl import RealTimePnLTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealTimeTradingEngine:
    """
    Complete real-time trading system

    Usage:
        engine = RealTimeTradingEngine(
            connector=AlpacaConnector(...),
            strategies={'momentum': MomentumStrategy(), ...},
            initial_capital=100000
        )

        await engine.start()
    """

    def __init__(
        self,
        connector,
        strategies: Dict[str, object],
        initial_capital: float = 100000.0,
        symbols: List[str] = None,
        update_interval: float = 1.0,  # seconds
        pnl_snapshot_interval: float = 60.0,  # seconds
    ):
        """
        Args:
            connector: Market data connector (e.g., AlpacaConnector)
            strategies: Dict of strategy instances
            initial_capital: Starting capital
            symbols: List of symbols to trade
            update_interval: How often to check for signals (seconds)
            pnl_snapshot_interval: How often to snapshot P&L
        """
        self.connector = connector
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL']
        self.update_interval = update_interval
        self.pnl_snapshot_interval = pnl_snapshot_interval

        # Initialize components
        self.strategy_router = AdaptiveStrategyRouter(strategies)
        self.pnl_tracker = RealTimePnLTracker(initial_capital=initial_capital)

        # Market data buffer
        self.market_data: Dict[str, MarketTick] = {}

        # Control flags
        self.is_running = False
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self.tick_count = 0
        self.signal_count = 0
        self.trade_count = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False

    async def start(self):
        """Start the trading engine"""
        logger.info("=" * 70)
        logger.info("STARTING REAL-TIME TRADING ENGINE")
        logger.info("=" * 70)
        logger.info(f"Initial Capital: ${self.pnl_tracker.initial_capital:,.2f}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Strategies: {', '.join(self.strategy_router.strategies.keys())}")
        logger.info("=" * 70)

        self.is_running = True

        try:
            # Register tick handler
            @self.connector.on_tick
            async def handle_tick(tick: MarketTick):
                await self._process_tick(tick)

            # Connect to data source
            await self.connector.start()
            await self.connector.subscribe(self.symbols)

            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._signal_generator_loop()),
                asyncio.create_task(self._pnl_snapshot_loop()),
                asyncio.create_task(self._status_reporter_loop()),
            ]

            # Wait for tasks to complete (or until shutdown)
            await asyncio.gather(*self._tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in trading engine: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping trading engine...")
        self.is_running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Close all positions
        await self._close_all_positions()

        # Disconnect
        await self.connector.stop()

        # Print final summary
        self._print_final_summary()

        logger.info("Trading engine stopped")

    async def _process_tick(self, tick: MarketTick):
        """Process incoming market tick"""
        self.tick_count += 1

        # Update market data buffer
        self.market_data[tick.symbol] = tick

        # Update P&L tracker with latest price
        self.pnl_tracker.update_market_price(tick.symbol, tick.price)

        # Log every 100 ticks
        if self.tick_count % 100 == 0:
            logger.info(f"Processed {self.tick_count} ticks, "
                       f"P&L: ${self.pnl_tracker.get_total_pnl():.2f}")

    async def _signal_generator_loop(self):
        """Main signal generation loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.update_interval)

                if not self.market_data:
                    continue

                # Process each symbol
                for symbol, tick in self.market_data.items():
                    await self._generate_signal(symbol, tick)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal generator: {e}")

    async def _generate_signal(self, symbol: str, tick: MarketTick):
        """Generate trading signal for a symbol"""
        try:
            # Prepare market data for strategy router
            market_data = {
                'symbol': symbol,
                'price': tick.price,
                'volume': tick.volume,
                'bid': tick.bid,
                'ask': tick.ask,
                'timestamp': tick.timestamp
            }

            # Select best strategy
            strategy_name, confidence = self.strategy_router.select_strategy(market_data)

            if confidence < 0.3:
                # Too low confidence, skip
                return

            # Get strategy instance
            strategy = self.strategy_router.strategies[strategy_name]

            # Generate signal (simplified - you'd call actual strategy method)
            signal = self._evaluate_strategy_signal(strategy, market_data)

            if signal != 0:
                self.signal_count += 1
                await self._execute_signal(symbol, signal, tick.price, strategy_name)

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")

    def _evaluate_strategy_signal(self, strategy, market_data: Dict) -> int:
        """
        Evaluate strategy and return signal
        Returns: 1 (buy), -1 (sell), 0 (hold)

        NOTE: This is simplified. In real implementation, you'd call:
        signal = strategy.generate_signal(market_data_history)
        """
        # Placeholder logic
        # In reality, you'd maintain price history and call strategy.generate_signal()

        # For now, return random signal based on confidence
        import random
        if random.random() > 0.95:  # 5% chance of signal
            return random.choice([-1, 1])
        return 0

    async def _execute_signal(self, symbol: str, signal: int, price: float, strategy_name: str):
        """Execute trading signal"""
        try:
            # Calculate position size (simple: 10% of capital per trade)
            capital = self.pnl_tracker.get_equity()
            position_size = capital * 0.1 / price

            if signal > 0:
                # Buy signal
                quantity = position_size
            else:
                # Sell signal (or close position)
                if symbol in self.pnl_tracker.positions:
                    # Close existing position
                    pnl = self.pnl_tracker.close_position(symbol, price, strategy_name)
                    if pnl is not None:
                        self.trade_count += 1
                        self.strategy_router.update_performance(strategy_name, pnl, datetime.now())
                        logger.info(f"Closed {symbol}: P&L = ${pnl:.2f}")
                    return
                else:
                    # Short signal (skip for now in simple version)
                    return

            # Execute trade
            trade = self.pnl_tracker.open_position(
                symbol=symbol,
                quantity=quantity,
                price=price,
                strategy=strategy_name
            )

            if trade:
                self.trade_count += 1
                logger.info(f"Executed {trade.side.upper()} {symbol}: "
                           f"{trade.quantity:.2f} @ ${trade.price:.2f} "
                           f"(Strategy: {strategy_name})")

        except Exception as e:
            logger.error(f"Error executing signal: {e}")

    async def _pnl_snapshot_loop(self):
        """Periodic P&L snapshot"""
        while self.is_running:
            try:
                await asyncio.sleep(self.pnl_snapshot_interval)
                self.pnl_tracker.snapshot()

                logger.info("-" * 70)
                logger.info(f"P&L Snapshot at {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"Total P&L: ${self.pnl_tracker.get_total_pnl():.2f}")
                logger.info(f"Equity: ${self.pnl_tracker.get_equity():.2f}")
                logger.info(f"Open Positions: {len(self.pnl_tracker.positions)}")
                logger.info("-" * 70)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in P&L snapshot: {e}")

    async def _status_reporter_loop(self):
        """Periodic status report"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                logger.info("\n" + "=" * 70)
                logger.info("STATUS REPORT")
                logger.info("=" * 70)

                # Performance metrics
                metrics = self.pnl_tracker.get_performance_metrics()
                for key, value in metrics.items():
                    logger.info(f"{key}: {value}")

                # Strategy performance
                logger.info("\nStrategy Performance:")
                perf_df = self.strategy_router.get_performance_summary()
                if not perf_df.empty:
                    logger.info(f"\n{perf_df.to_string(index=False)}")

                # Market regime
                regime = self.strategy_router.get_regime_info()
                logger.info(f"\nMarket Regime: {regime.get('regime', 'unknown')}")
                logger.info(f"Confidence: {regime.get('confidence', 'N/A')}")

                logger.info("=" * 70 + "\n")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status reporter: {e}")

    async def _close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")

        for symbol in list(self.pnl_tracker.positions.keys()):
            if symbol in self.market_data:
                price = self.market_data[symbol].price
                pnl = self.pnl_tracker.close_position(symbol, price, strategy="close_all")
                if pnl is not None:
                    logger.info(f"Closed {symbol}: P&L = ${pnl:.2f}")

    def _print_final_summary(self):
        """Print final trading summary"""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL TRADING SUMMARY")
        logger.info("=" * 70)

        # Statistics
        logger.info(f"Total Ticks Processed: {self.tick_count}")
        logger.info(f"Total Signals Generated: {self.signal_count}")
        logger.info(f"Total Trades Executed: {self.trade_count}")

        # P&L
        self.pnl_tracker.print_summary()

        # Strategy performance
        logger.info("Strategy Performance:")
        perf_df = self.strategy_router.get_performance_summary()
        if not perf_df.empty:
            print(perf_df.to_string(index=False))

        logger.info("=" * 70)


# Example usage
if __name__ == "__main__":
    # This is a demo - you'd import real connectors and strategies
    from connectors.alpaca_connector import AlpacaConnector
    from strategy.classical import MomentumStrategy
    from strategy.hft_strategies import MarketMakingStrategy

    async def main():
        # Initialize connector
        connector = AlpacaConnector(
            api_key="YOUR_API_KEY",
            api_secret="YOUR_API_SECRET",
            paper=True
        )

        # Initialize strategies
        strategies = {
            'momentum': MomentumStrategy(),
            'market_making': MarketMakingStrategy(),
        }

        # Create engine
        engine = RealTimeTradingEngine(
            connector=connector,
            strategies=strategies,
            initial_capital=100000.0,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            update_interval=5.0,  # Check for signals every 5 seconds
        )

        # Start trading
        await engine.start()

    # Run
    asyncio.run(main())

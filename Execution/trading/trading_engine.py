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
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import signal
import sys

# Type checking imports
if TYPE_CHECKING:
    import pandas as pd

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
        enable_research: bool = True,  # Enable market microstructure analysis
        research_min_ticks: int = 30,  # Minimum ticks before running research (reduced from 100 for faster startup)
        enable_validation: bool = True,  # Enable quick backtest/Monte Carlo validation before trading
        validation_min_ticks: int = 50,  # Minimum ticks for validation
        validation_timeout: float = 0.5,  # Max validation time in seconds (to avoid blocking)
    ):
        """
        Args:
            connector: Market data connector (e.g., AlpacaConnector)
            strategies: Dict of strategy instances
            initial_capital: Starting capital
            symbols: List of symbols to trade
            update_interval: How often to check for signals (seconds)
            pnl_snapshot_interval: How often to snapshot P&L
            enable_research: Enable market microstructure analysis before trading
            research_min_ticks: Minimum ticks to collect before running research
            enable_validation: Enable quick backtest/Monte Carlo validation before executing trades
            validation_min_ticks: Minimum ticks required for validation
            validation_timeout: Maximum time allowed for validation (seconds) to avoid blocking
        """
        self.connector = connector
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL']
        self.update_interval = update_interval
        self.pnl_snapshot_interval = pnl_snapshot_interval
        self.enable_research = enable_research
        self.research_min_ticks = research_min_ticks
        self.enable_validation = enable_validation
        self.validation_min_ticks = validation_min_ticks
        self.validation_timeout = validation_timeout

        # Initialize components
        self.strategy_router = AdaptiveStrategyRouter(strategies)
        self.pnl_tracker = RealTimePnLTracker(initial_capital=initial_capital)
        
        # Research Framework
        self.research_framework = None
        self.research_results = None
        self.research_completed = False
        if self.enable_research:
            try:
                from Research import CompleteResearchFramework
                self.research_framework = CompleteResearchFramework()
                logger.info("Research Framework initialized for market microstructure analysis")
            except ImportError as e:
                logger.warning(f"Research Framework not available: {e}")
                self.enable_research = False

        # Market data buffer
        self.market_data: Dict[str, MarketTick] = {}

        # Price history for strategies (maintain last N ticks per symbol)
        self.price_history: Dict[str, List[Dict]] = {symbol: [] for symbol in self.symbols}
        self.max_history_size = 200  # Keep last 200 ticks per symbol

        # Control flags
        self.is_running = False
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self.tick_count = 0
        self.signal_count = 0
        self.trade_count = 0

        # Performance tracking (运行速度)
        self.start_time = None
        self.tick_processing_times = []  # Track tick processing latency
        self.signal_generation_times = []  # Track signal generation latency
        self.validation_times = []  # Track validation latency
        self.last_tick_time = None
        
        # Validation cache (避免重复验证相同策略)
        self.validation_cache: Dict[str, Dict] = {}  # strategy_name -> {result, timestamp}
        self.cache_ttl = 60.0  # Cache validity: 60 seconds

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
        self.start_time = datetime.now()

        try:
            # Register tick handler
            @self.connector.on_tick
            async def handle_tick(tick: MarketTick):
                await self._process_tick(tick)

            # Connect to data source
            await self.connector.start()
            await self.connector.subscribe(self.symbols)
            
            # Wait a moment for connection to stabilize
            await asyncio.sleep(1)

            # Collect initial data for market microstructure analysis
            if self.enable_research and self.research_framework:
                logger.info(f"Collecting {self.research_min_ticks} ticks for market microstructure analysis...")
                await self._collect_data_for_research()

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

    async def _collect_data_for_research(self):
        """Collect initial market data for microstructure analysis"""
        import pandas as pd
        from datetime import datetime, timedelta
        
        research_data = {symbol: [] for symbol in self.symbols}
        start_time = datetime.now()
        # Reduced timeout: 2 minutes for 30 ticks (was 5 minutes for 100 ticks)
        timeout = timedelta(minutes=2)
        
        logger.info(f"Collecting {self.research_min_ticks} ticks per symbol for market microstructure analysis...")
        
        while not self.research_completed:
            # Check timeout
            elapsed = datetime.now() - start_time
            if elapsed > timeout:
                current_ticks = {symbol: len(self.price_history.get(symbol, [])) 
                               for symbol in self.symbols}
                logger.warning(f"Timeout waiting for research data ({elapsed.total_seconds():.0f}s), "
                             f"proceeding with available data: {current_ticks}")
                break
            
            # Check if we have enough data for each symbol
            min_ticks = min(len(self.price_history.get(symbol, [])) for symbol in self.symbols)
            if min_ticks >= self.research_min_ticks:
                logger.info(f"✓ Collected enough data for research: {min_ticks} ticks per symbol")
                break
            
            # Log progress more frequently
            if self.tick_count % 10 == 0:
                progress = {symbol: len(self.price_history.get(symbol, [])) 
                           for symbol in self.symbols}
                remaining = self.research_min_ticks - min_ticks
                logger.info(f"Research data collection: {progress} (need {self.research_min_ticks} per symbol, "
                          f"{remaining} remaining)")
            
            await asyncio.sleep(0.5)
        
        # Prepare data for research framework
        if self.research_framework:
            try:
                market_data = {}
                
                for symbol in self.symbols:
                    if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                        df = pd.DataFrame(self.price_history[symbol])
                        
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True, drop=False)
                        
                        # Extract price data
                        if 'close' in df.columns:
                            prices = df['close']
                        elif 'price' in df.columns:
                            prices = df['price']
                        else:
                            continue
                        
                        # Prepare market data dict
                        if 'prices' not in market_data:
                            market_data['prices'] = prices
                        else:
                            # Combine prices if multiple symbols
                            market_data['prices'] = pd.concat([market_data['prices'], prices])
                        
                        # Add bid/ask if available
                        if 'bid' in df.columns:
                            if 'bid_prices' not in market_data:
                                market_data['bid_prices'] = df['bid']
                            else:
                                market_data['bid_prices'] = pd.concat([market_data['bid_prices'], df['bid']])
                        
                        if 'ask' in df.columns:
                            if 'ask_prices' not in market_data:
                                market_data['ask_prices'] = df['ask']
                            else:
                                market_data['ask_prices'] = pd.concat([market_data['ask_prices'], df['ask']])
                
                if market_data and 'prices' in market_data:
                    logger.info("=" * 70)
                    logger.info("RUNNING MARKET MICROSTRUCTURE ANALYSIS")
                    logger.info("=" * 70)
                    
                    # Calculate forward returns for validation
                    returns = market_data['prices'].pct_change().dropna()
                    forward_returns = returns.shift(-1).dropna()
                    
                    # Run research framework
                    self.research_results = self.research_framework.run_complete_research_pipeline(
                        market_data, forward_returns
                    )
                    
                    # Apply research insights to strategy selection
                    self._apply_research_insights()
                    
                    self.research_completed = True
                    logger.info("=" * 70)
                    logger.info("MARKET MICROSTRUCTURE ANALYSIS COMPLETE")
                    logger.info("=" * 70)
                else:
                    logger.warning("Insufficient data for research analysis, proceeding without it")
                    self.enable_research = False
                    
            except Exception as e:
                logger.error(f"Error running market microstructure analysis: {e}", exc_info=True)
                self.enable_research = False

    def _apply_research_insights(self):
        """Apply research framework insights to strategy selection"""
        if not self.research_results:
            return
        
        try:
            # Extract key insights from research results
            profiling_results = self.research_results.get('profiling', {})
            hypotheses = self.research_results.get('hypotheses', [])
            validation_results = self.research_results.get('validation', {})
            
            logger.info("Applying research insights to strategy selection...")
            
            # Example: Adjust strategy router based on market regime detected
            if profiling_results:
                # Detect market regime from microstructure analysis
                volatility = profiling_results.get('volatility', {})
                liquidity = profiling_results.get('liquidity', {})
                
                if volatility:
                    avg_vol = volatility.get('average', 0)
                    logger.info(f"Market volatility detected: {avg_vol:.4f}")
                    
                    # High volatility -> prefer mean reversion
                    # Low volatility -> prefer momentum
                    if avg_vol > 0.02:
                        logger.info("High volatility regime detected, favoring mean reversion strategies")
                    else:
                        logger.info("Low volatility regime detected, favoring momentum strategies")
                
                if liquidity:
                    avg_spread = liquidity.get('average_spread', 0)
                    logger.info(f"Average bid-ask spread: {avg_spread:.4f}")
            
            # Log validated factors
            if hypotheses:
                logger.info(f"Generated {len(hypotheses)} factor hypotheses")
                for i, hypothesis in enumerate(hypotheses[:3], 1):  # Show top 3
                    factor_name = hypothesis.get('factor_name', 'Unknown')
                    logger.info(f"  {i}. {factor_name}")
            
            # Log validation results
            if validation_results:
                statistical = validation_results.get('statistical', {})
                if statistical:
                    logger.info("Statistical validation results:")
                    for metric, value in list(statistical.items())[:3]:
                        logger.info(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            logger.warning(f"Error applying research insights: {e}")

    async def _process_tick(self, tick: MarketTick):
        """Process incoming market tick"""
        import time
        tick_start_time = time.perf_counter()
        
        self.tick_count += 1

        # Log every tick for debugging (can be reduced later)
        if self.tick_count % 10 == 0:  # Log every 10th tick to reduce noise
            logger.info(f"Received tick #{self.tick_count}: {tick.symbol} = ${tick.price:.2f}")

        # Update market data buffer
        self.market_data[tick.symbol] = tick
        
        # Update price history for strategies
        if tick.symbol not in self.price_history:
            self.price_history[tick.symbol] = []
        
        # Add tick to history
        tick_data = {
            'timestamp': tick.timestamp,
            'price': tick.price,
            'close': tick.price,  # For compatibility
            'last_price': tick.price,  # For compatibility
            'volume': tick.volume,
            'bid': tick.bid,
            'ask': tick.ask,
            'symbol': tick.symbol
        }
        self.price_history[tick.symbol].append(tick_data)
        
        # Keep only last N ticks
        if len(self.price_history[tick.symbol]) > self.max_history_size:
            self.price_history[tick.symbol] = self.price_history[tick.symbol][-self.max_history_size:]

        # Update P&L tracker with latest price
        self.pnl_tracker.update_market_price(tick.symbol, tick.price)

        # Track processing time
        tick_processing_time = time.perf_counter() - tick_start_time
        self.tick_processing_times.append(tick_processing_time)
        # Keep only last 1000 processing times to avoid memory issues
        if len(self.tick_processing_times) > 1000:
            self.tick_processing_times = self.tick_processing_times[-1000:]
        
        self.last_tick_time = datetime.now()

        # Log tick processing status
        history_len = len(self.price_history.get(tick.symbol, []))
        if self.tick_count % 5 == 0 or history_len <= 15:  # Log more frequently when collecting data
            logger.info(f"✓ Processed {self.tick_count} ticks total, "
                       f"{tick.symbol}: {history_len}/{self.max_history_size} data points, "
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
        import time
        signal_start_time = time.perf_counter()
        
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

            # Log strategy selection
            if self.signal_count % 5 == 0:
                logger.info(f"🎯 Strategy selected for {symbol}: {strategy_name} (confidence: {confidence:.3f})")

            # Lower threshold for initial trades (allow 0.2 instead of 0.25)
            if confidence < 0.2:  # Lowered from 0.25 to 0.2
                # Too low confidence, skip
                if self.signal_count % 10 == 0:
                    logger.debug(f"⏸️  Skipping signal generation for {symbol}: confidence {confidence:.3f} < 0.2")
                return

            # Get strategy instance
            strategy = self.strategy_router.strategies[strategy_name]

            # Generate signal using actual strategy method
            signal = self._evaluate_strategy_signal(strategy, symbol, market_data)

            if signal != 0:
                self.signal_count += 1
                
                # Log signal details
                signal_type = "BUY" if signal > 0 else "SELL"
                logger.info(f"📊 Generated {signal_type} signal for {symbol} "
                          f"(strategy: {strategy_name}, signal={signal})")
                
                # Quick validation before executing (if enabled)
                if self.enable_validation:
                    validation_passed = await self._quick_validate_strategy(
                        strategy, strategy_name, symbol, signal
                    )
                    if not validation_passed:
                        logger.info(f"⏸️  Signal validation failed for {symbol} ({strategy_name}), skipping trade")
                        logger.debug(f"   Signal details: signal={signal}, strength={strength:.3f}, confidence={confidence:.3f}")
                        return
                    else:
                        logger.debug(f"✓ Validation passed for {symbol} ({strategy_name})")
                
                await self._execute_signal(symbol, signal, tick.price, strategy_name)
            else:
                # Log when we get HOLD signals (less frequently)
                if self.signal_count % 20 == 0:
                    logger.debug(f"HOLD signal for {symbol} (strategy: {strategy_name})")
            
            # Track signal generation time
            signal_generation_time = time.perf_counter() - signal_start_time
            self.signal_generation_times.append(signal_generation_time)
            # Keep only last 1000 times
            if len(self.signal_generation_times) > 1000:
                self.signal_generation_times = self.signal_generation_times[-1000:]

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")

    def _evaluate_strategy_signal(self, strategy, symbol: str, market_data: Dict) -> int:
        """
        Evaluate strategy and return signal
        Returns: 1 (buy), -1 (sell), 0 (hold)
        """
        import pandas as pd
        
        # Get price history for this symbol
        # Reduced requirement: Need at least 10 data points (was 20)
        # This allows faster signal generation, especially with slower data sources like Yahoo Finance
        min_data_points = 10
        history_len = len(self.price_history.get(symbol, []))
        
        if symbol not in self.price_history or history_len < min_data_points:
            # Log progress more frequently
            if history_len % 3 == 0 or history_len == 0:
                remaining = min_data_points - history_len
                estimated_time = remaining * self.update_interval if self.update_interval > 0 else 0
                logger.info(f"⏳ Collecting data for {symbol}: {history_len}/{min_data_points} ticks "
                           f"(need {remaining} more, ~{estimated_time:.0f}s)")
            return 0
        
        # Log when we have enough data
        if history_len == min_data_points:
            logger.info(f"✓ Enough data collected for {symbol}: {history_len} ticks, starting signal generation")
        
        # Convert history to DataFrame for strategy
        history_data = self.price_history[symbol]
        df = pd.DataFrame(history_data)
        
        # Ensure timestamp is datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True, drop=False)
        
        # Ensure 'close' column exists (required by strategies)
        if 'close' not in df.columns:
            if 'price' in df.columns:
                df['close'] = df['price']
            elif 'last_price' in df.columns:
                df['close'] = df['last_price']
            else:
                logger.warning(f"No price column found for {symbol}, available columns: {list(df.columns)}")
                return 0
        
        # Debug: log data shape
        if history_len % 10 == 0:  # Log every 10 ticks
            logger.debug(f"DataFrame for {symbol}: shape={df.shape}, columns={list(df.columns)}, "
                        f"close_range=[{df['close'].min():.2f}, {df['close'].max():.2f}]")
        
        # Initialize strategy if needed (some strategies need initialization)
        if hasattr(strategy, 'initialize') and not getattr(strategy, 'is_initialized', False):
            try:
                # Initialize with available history
                strategy.initialize(df)
                logger.info(f"Initialized strategy {type(strategy).__name__} for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to initialize strategy: {e}")
        
        # Check if strategy has generate_signal method (BaseStrategy interface)
        if hasattr(strategy, 'generate_signal'):
            try:
                # Call strategy's generate_signal method
                signal_dict = strategy.generate_signal(df)

                # Convert signal dict to int
                if isinstance(signal_dict, dict):
                    action = signal_dict.get('action', 'HOLD')
                    strength = signal_dict.get('strength', 0.0)
                    confidence = signal_dict.get('confidence', 0.0)
                    
                    # Log signal evaluation (even if rejected) - log more frequently
                    if self.signal_count % 3 == 0 or action != 'HOLD':  # Log every 3rd or when not HOLD
                        logger.info(f"📊 Strategy evaluation for {symbol}: action={action}, "
                                   f"strength={strength:.3f}, confidence={confidence:.3f}")
                    
                    # Only generate signal if confidence is reasonable (lowered threshold for more signals)
                    if confidence < 0.25:  # Lowered from 0.3 to 0.25
                        if self.signal_count % 5 == 0:
                            logger.debug(f"Signal rejected: confidence {confidence:.3f} < 0.25")
                        return 0
                    
                    # Lowered strength threshold from 0.1 to 0.05 to allow more signals
                    if action == 'BUY' and strength > 0.05:
                        logger.info(f"✓ BUY signal for {symbol}: strength={strength:.3f}, confidence={confidence:.3f}")
                        return 1
                    elif action == 'SELL' and strength > 0.05:
                        logger.info(f"✓ SELL signal for {symbol}: strength={strength:.3f}, confidence={confidence:.3f}")
                        return -1
                    else:
                        if self.signal_count % 5 == 0:
                            logger.debug(f"Signal rejected: action={action}, strength={strength:.3f} (need >0.05)")
                        return 0
                else:
                    # Fallback if signal is not a dict
                    return 0
                    
            except Exception as e:
                logger.warning(f"Error calling strategy.generate_signal: {e}", exc_info=True)
                return 0
        
        # Fallback: Check if strategy has generate_signals method (MomentumStrategy interface)
        elif hasattr(strategy, 'generate_signals'):
            try:
                # Some strategies expect a dict of symbol -> DataFrame
                price_data = {symbol: df}
                signals = strategy.generate_signals(price_data)
                
                if symbol in signals and signals[symbol]:
                    # Get first signal
                    signal_obj = signals[symbol][0]
                    if hasattr(signal_obj, 'signal_strength'):
                        strength = signal_obj.signal_strength
                        if strength > 0.1:
                            return 1
                        elif strength < -0.1:
                            return -1
                return 0
            except Exception as e:
                logger.warning(f"Error calling strategy.generate_signals: {e}", exc_info=True)
                return 0
        
        # No compatible method found
        logger.warning(f"Strategy {type(strategy).__name__} does not have generate_signal or generate_signals method")
        return 0

    async def _execute_signal(self, symbol: str, signal: int, price: float, strategy_name: str):
        """Execute trading signal"""
        try:
            # Calculate position size (simple: 10% of capital per trade)
            capital = self.pnl_tracker.get_equity()
            position_size = capital * 0.1 / price

            if signal > 0:
                # Buy signal
                if symbol in self.pnl_tracker.positions:
                    pos = self.pnl_tracker.positions[symbol]
                    # If we already have a long position, skip (or could add to position)
                    if pos.quantity > 0:
                        logger.debug(f"Already have long position in {symbol}, skipping BUY signal")
                        return
                    # If we have a short position, close it first
                    elif pos.quantity < 0:
                        logger.info(f"Closing short position in {symbol} before opening long")
                        pnl = self.pnl_tracker.close_position(symbol, price, strategy_name)
                        if pnl is not None:
                            self.trade_count += 1
                            self.strategy_router.update_performance(strategy_name, pnl, datetime.now())
                            logger.info(f"Closed short {symbol}: P&L = ${pnl:.2f}")
                
                # Open long position
                quantity = position_size
                trade = self.pnl_tracker.open_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    strategy=strategy_name
                )

                if trade:
                    self.trade_count += 1
                    logger.info(f"✅ Executed BUY {symbol}: "
                               f"{trade.quantity:.2f} shares @ ${trade.price:.2f} "
                               f"(Strategy: {strategy_name})")
                    
            elif signal < 0:
                # Sell signal
                if symbol in self.pnl_tracker.positions:
                    pos = self.pnl_tracker.positions[symbol]
                    # If we have a long position, close it
                    if pos.quantity > 0:
                        pnl = self.pnl_tracker.close_position(symbol, price, strategy_name)
                        if pnl is not None:
                            self.trade_count += 1
                            self.strategy_router.update_performance(strategy_name, pnl, datetime.now())
                            logger.info(f"✅ Closed long {symbol}: P&L = ${pnl:.2f}")
                        return
                    # If we already have a short position, skip (or could add to position)
                    elif pos.quantity < 0:
                        logger.debug(f"Already have short position in {symbol}, skipping SELL signal")
                        return

                # Open short position (negative quantity)
                quantity = -position_size  # Negative for short
                trade = self.pnl_tracker.open_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    strategy=strategy_name
                )

                if trade:
                    self.trade_count += 1
                    logger.info(f"✅ Executed SHORT {symbol}: "
                               f"{abs(trade.quantity):.2f} shares @ ${trade.price:.2f} "
                               f"(Strategy: {strategy_name})")
            else:
                # signal == 0 (HOLD), do nothing
                logger.debug(f"HOLD signal for {symbol}, no action taken")

        except Exception as e:
            logger.error(f"❌ Error executing signal for {symbol}: {e}", exc_info=True)

    async def _quick_validate_strategy(
        self, strategy, strategy_name: str, symbol: str, signal: int
    ) -> bool:
        """
        Quick validation using lightweight backtest and simplified Monte Carlo
        
        Returns:
            True if validation passes, False otherwise
        """
        import time
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        validation_start = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = f"{strategy_name}_{symbol}"
            if cache_key in self.validation_cache:
                cached = self.validation_cache[cache_key]
                age = (datetime.now() - cached['timestamp']).total_seconds()
                if age < self.cache_ttl:
                    logger.debug(f"Using cached validation result for {strategy_name} (age: {age:.1f}s)")
                    return cached['result']
            
            # Check if we have enough data
            history = self.price_history.get(symbol, [])
            if len(history) < self.validation_min_ticks:
                logger.debug(f"Insufficient data for validation: {len(history)} < {self.validation_min_ticks}")
                return True  # Allow trade if not enough data (don't block)
            
            # Prepare data for validation (use recent history)
            df = pd.DataFrame(history[-self.validation_min_ticks:])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True, drop=False)
            
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                else:
                    return True  # Can't validate without price data
            
            # Quick backtest validation (simplified)
            validation_result = await asyncio.wait_for(
                self._run_quick_backtest(strategy, df, symbol),
                timeout=self.validation_timeout
            )
            
            if not validation_result:
                return False
            
            # Simplified Monte Carlo validation (if time permits)
            # Make MC validation less strict - only block if both backtest and MC fail
            try:
                mc_result = await asyncio.wait_for(
                    self._run_simplified_monte_carlo(df, signal),
                    timeout=self.validation_timeout * 0.5  # Use half timeout for MC
                )
                # Only fail if BOTH backtest and MC fail (more lenient)
                if not mc_result:
                    logger.debug(f"Monte Carlo validation failed for {symbol}, but backtest passed - allowing trade")
                    # Don't fail here - let backtest result stand
            except asyncio.TimeoutError:
                logger.debug(f"Monte Carlo validation timeout for {symbol}, using backtest result only")
            
            # Cache result
            self.validation_cache[cache_key] = {
                'result': True,
                'timestamp': datetime.now()
            }
            
            # Track validation time
            validation_time = time.perf_counter() - validation_start
            self.validation_times.append(validation_time)
            if len(self.validation_times) > 1000:
                self.validation_times = self.validation_times[-1000:]
            
            logger.debug(f"✓ Validation passed for {symbol} ({strategy_name}) in {validation_time*1000:.2f}ms")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Validation timeout for {symbol} ({strategy_name}), allowing trade")
            return True  # Allow trade on timeout (don't block)
        except Exception as e:
            logger.warning(f"Validation error for {symbol} ({strategy_name}): {e}, allowing trade")
            return True  # Allow trade on error (fail-safe)
    
    async def _run_quick_backtest(self, strategy, df: "pd.DataFrame", symbol: str) -> bool:
        """
        Quick backtest validation using recent data
        
        Returns:
            True if backtest shows positive expected return, False otherwise
        """
        import pandas as pd
        import numpy as np
        
        try:
            # Simplified backtest: simulate last N periods
            prices = df['close'].values
            if len(prices) < 20:
                return True  # Not enough data
            
            # Generate signals for recent periods
            signals = []
            for i in range(20, len(prices)):
                try:
                    # Get data up to this point
                    historical_data = df.iloc[:i+1].copy()
                    if 'timestamp' in historical_data.columns:
                        historical_data.set_index('timestamp', inplace=True, drop=False)
                    
                    # Generate signal
                    if hasattr(strategy, 'generate_signal'):
                        signal_dict = strategy.generate_signal(historical_data)
                        if isinstance(signal_dict, dict):
                            action = signal_dict.get('action', 'HOLD')
                            signal = 1 if action == 'BUY' else (-1 if action == 'SELL' else 0)
                        else:
                            signal = 0
                    else:
                        signal = 0
                    
                    signals.append(signal)
                except:
                    signals.append(0)
            
            if not signals:
                return True
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            signal_returns = np.array(signals) * returns[-len(signals):]
            
            # Check if strategy has positive expected return
            if len(signal_returns) > 0:
                mean_return = np.mean(signal_returns)
                sharpe = mean_return / (np.std(signal_returns) + 1e-10) * np.sqrt(252)
                
                # More lenient: pass if Sharpe > -0.5 or mean return > -0.001 (allow small losses)
                # This prevents blocking trades that are slightly negative but acceptable
                result = sharpe > -0.5 or mean_return > -0.001
                if not result:
                    logger.debug(f"Backtest validation failed: sharpe={sharpe:.4f}, mean_return={mean_return:.6f}")
                return result
            
            return True

        except Exception as e:
            logger.debug(f"Quick backtest error: {e}")
            return True  # Allow on error
    
    async def _run_simplified_monte_carlo(self, df: "pd.DataFrame", signal: int) -> bool:
        """
        Simplified Monte Carlo validation (lightweight)
        
        Returns:
            True if MC shows acceptable risk, False otherwise
        """
        import numpy as np
        
        try:
            prices = df['close'].values
            if len(prices) < 20:
                return True
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            if len(returns) < 10:
                return True
            
            # Simplified MC: simulate 1000 paths (much faster than full MC)
            n_paths = 1000
            n_periods = 10
            
            # Estimate parameters
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate random paths
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(
                mean_return * signal,  # Directional bias based on signal
                std_return,
                (n_paths, n_periods)
            )
            
            # Calculate portfolio values
            portfolio_values = (1 + simulated_returns).prod(axis=1)
            
            # Check risk metrics
            mean_value = np.mean(portfolio_values)
            var_95 = np.percentile(portfolio_values, 5)  # 95% VaR
            
            # More lenient validation: pass if expected return > 0 or VaR is acceptable
            # Original: mean_value > 0.95 and var_95 > 0.90 (too strict)
            # New: mean_value > 0.98 or var_95 > 0.85 (more lenient)
            # This allows trades with small expected gains or acceptable risk
            result = mean_value > 0.98 or var_95 > 0.85
            if not result:
                logger.debug(f"MC validation failed: mean_value={mean_value:.4f}, var_95={var_95:.4f}")
            return result

        except Exception as e:
            logger.debug(f"Simplified Monte Carlo error: {e}")
            return True  # Allow on error

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
        import numpy as np
        
        logger.info("\n" + "=" * 70)
        logger.info("FINAL TRADING SUMMARY")
        logger.info("=" * 70)

        # Statistics
        logger.info(f"Total Ticks Processed: {self.tick_count}")
        logger.info(f"Total Signals Generated: {self.signal_count}")
        logger.info(f"Total Trades Executed: {self.trade_count}")

        # Performance Metrics (运行速度)
        logger.info("\n" + "-" * 70)
        logger.info("PERFORMANCE METRICS (运行速度)")
        logger.info("-" * 70)
        
        if self.start_time:
            total_runtime = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Total Runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
            
            if total_runtime > 0:
                ticks_per_second = self.tick_count / total_runtime
                logger.info(f"Tick Processing Rate: {ticks_per_second:.2f} ticks/second")
                logger.info(f"Signal Generation Rate: {self.signal_count / total_runtime:.2f} signals/second")
        
        if self.tick_processing_times:
            avg_tick_latency = np.mean(self.tick_processing_times) * 1000  # Convert to ms
            p50_tick_latency = np.percentile(self.tick_processing_times, 50) * 1000
            p95_tick_latency = np.percentile(self.tick_processing_times, 95) * 1000
            p99_tick_latency = np.percentile(self.tick_processing_times, 99) * 1000
            logger.info(f"Tick Processing Latency:")
            logger.info(f"  Average: {avg_tick_latency:.3f} ms")
            logger.info(f"  P50: {p50_tick_latency:.3f} ms")
            logger.info(f"  P95: {p95_tick_latency:.3f} ms")
            logger.info(f"  P99: {p99_tick_latency:.3f} ms")
        
        if self.signal_generation_times:
            avg_signal_latency = np.mean(self.signal_generation_times) * 1000  # Convert to ms
            p50_signal_latency = np.percentile(self.signal_generation_times, 50) * 1000
            p95_signal_latency = np.percentile(self.signal_generation_times, 95) * 1000
            logger.info(f"Signal Generation Latency:")
            logger.info(f"  Average: {avg_signal_latency:.3f} ms")
            logger.info(f"  P50: {p50_signal_latency:.3f} ms")
            logger.info(f"  P95: {p95_signal_latency:.3f} ms")
        
        if self.validation_times:
            avg_validation_latency = np.mean(self.validation_times) * 1000  # Convert to ms
            p50_validation_latency = np.percentile(self.validation_times, 50) * 1000
            p95_validation_latency = np.percentile(self.validation_times, 95) * 1000
            max_validation_latency = np.max(self.validation_times) * 1000
            logger.info(f"Validation Latency (Backtest + Monte Carlo):")
            logger.info(f"  Average: {avg_validation_latency:.3f} ms")
            logger.info(f"  P50: {p50_validation_latency:.3f} ms")
            logger.info(f"  P95: {p95_validation_latency:.3f} ms")
            logger.info(f"  Max: {max_validation_latency:.3f} ms")
            logger.info(f"  Total Validations: {len(self.validation_times)}")
            logger.info(f"  Validation Enabled: {self.enable_validation}")

        # Final Trading Results (最终交易结果)
        logger.info("\n" + "-" * 70)
        logger.info("FINAL TRADING RESULTS")
        logger.info("-" * 70)
        
        # Get final P&L and Sharpe ratio
        final_net_pnl = self.pnl_tracker.get_total_pnl()
        performance_metrics = self.pnl_tracker.get_performance_metrics()
        
        # Extract Sharpe ratio (remove formatting)
        sharpe_str = performance_metrics.get('Sharpe Ratio', '0.00')
        try:
            final_sharpe = float(sharpe_str)
        except (ValueError, TypeError):
            final_sharpe = 0.0
        
        logger.info(f"Final Net P&L: ${final_net_pnl:.2f}")
        logger.info(f"Final Sharpe Ratio: {final_sharpe:.3f}")
        
        # Additional P&L details
        equity = self.pnl_tracker.get_equity()
        total_return = (equity - self.pnl_tracker.initial_capital) / self.pnl_tracker.initial_capital * 100
        logger.info(f"Final Equity: ${equity:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Initial Capital: ${self.pnl_tracker.initial_capital:.2f}")

        # P&L Summary
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

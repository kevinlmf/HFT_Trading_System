"""
Realistic Market Simulator
Simulates real market conditions including spreads, slippage, latency, and partial fills
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class MarketConditions:
    """Current market conditions affecting execution"""
    volatility: float  # Current volatility level
    volume_profile: float  # Relative volume (1.0 = normal)
    spread_multiplier: float  # Spread widening factor
    liquidity_factor: float  # Market liquidity (1.0 = normal)
    market_impact_factor: float  # Price impact sensitivity
    latency_ms: float  # Network/processing latency in ms

@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    timestamp: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Fill:
    """Trade execution fill"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    market_impact: float

class RealisticMarketSimulator:
    """
    Simulates realistic market conditions for backtesting
    
    Features:
    - Dynamic bid-ask spreads
    - Volume-based slippage
    - Latency simulation
    - Partial fills
    - Market impact modeling
    - Commission structure
    """
    
    def __init__(self, 
                 base_commission: float = 0.0005,  # 0.05% commission
                 min_commission: float = 1.0,      # Minimum $1 commission
                 base_spread_bps: Dict[str, float] = None,  # Base spreads by asset class
                 latency_range_ms: Tuple[float, float] = (10, 100),
                 market_impact_coefficient: float = 0.1,
                 liquidity_simulation: bool = True):
        """
        Initialize market simulator
        
        Args:
            base_commission: Base commission rate
            min_commission: Minimum commission per trade
            base_spread_bps: Base spreads in basis points by asset class
            latency_range_ms: Range of latency in milliseconds
            market_impact_coefficient: Coefficient for market impact calculation
            liquidity_simulation: Whether to simulate liquidity constraints
        """
        self.base_commission = base_commission
        self.min_commission = min_commission
        self.latency_range_ms = latency_range_ms
        self.market_impact_coefficient = market_impact_coefficient
        self.liquidity_simulation = liquidity_simulation
        
        # Default spreads by asset class (in basis points)
        self.base_spread_bps = base_spread_bps or {
            'equity': 5,      # 0.05%
            'forex': 2,       # 0.02%
            'crypto': 20,     # 0.2%
            'futures': 1,     # 0.01%
            'bonds': 3,       # 0.03%
            'commodities': 10 # 0.1%
        }
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fill_history: List[Fill] = []
        
        # Market state
        self.current_conditions = MarketConditions(
            volatility=1.0,
            volume_profile=1.0,
            spread_multiplier=1.0,
            liquidity_factor=1.0,
            market_impact_factor=1.0,
            latency_ms=50.0
        )
    
    def update_market_conditions(self, market_data: pd.DataFrame, 
                               current_timestamp: datetime):
        """
        Update market conditions based on current market data
        
        Args:
            market_data: Current market data with OHLCV and indicators
            current_timestamp: Current simulation timestamp
        """
        # Calculate volatility from recent price movements
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1]
            long_term_vol = returns.rolling(60).std().iloc[-1] if len(returns) >= 60 else current_vol
            self.current_conditions.volatility = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Volume profile assessment
        if 'volume' in market_data.columns and len(market_data) >= 20:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            self.current_conditions.volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Market hours effect on liquidity and spreads
        hour = current_timestamp.hour
        if 9 <= hour <= 16:  # Market hours (simplified)
            liquidity_multiplier = 1.0
            spread_multiplier = 1.0
        elif hour in [9, 16]:  # Opening/closing
            liquidity_multiplier = 0.7
            spread_multiplier = 1.5
        else:  # After hours
            liquidity_multiplier = 0.3
            spread_multiplier = 3.0
        
        self.current_conditions.liquidity_factor = liquidity_multiplier
        self.current_conditions.spread_multiplier = spread_multiplier * (1 + self.current_conditions.volatility * 0.5)
        
        # Dynamic latency based on market stress
        stress_factor = max(self.current_conditions.volatility, 1/self.current_conditions.liquidity_factor)
        base_latency = np.mean(self.latency_range_ms)
        self.current_conditions.latency_ms = base_latency * (0.5 + stress_factor)
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """
        Place a new order
        
        Returns:
            order_id: Unique identifier for the order
        """
        order_id = f"ORD_{len(self.pending_orders)}_{int(datetime.now().timestamp())}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        # Basic order validation
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return order_id
        
        self.pending_orders[order_id] = order
        return order_id
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        if order.quantity <= 0:
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False
        
        return True
    
    def process_orders(self, current_market_data: Dict[str, float]) -> List[Fill]:
        """
        Process pending orders against current market data
        
        Args:
            current_market_data: Dict of symbol -> current price
            
        Returns:
            List of fills generated
        """
        fills = []
        completed_orders = []
        
        # Simulate latency delay
        processing_delay = random.uniform(*self.latency_range_ms) / 1000.0  # Convert to seconds
        
        for order_id, order in self.pending_orders.items():
            if order.symbol not in current_market_data:
                continue
            
            current_price = current_market_data[order.symbol]
            
            # Check if order should be triggered/filled
            if self._should_fill_order(order, current_price):
                fill = self._execute_order(order, current_price)
                if fill:
                    fills.append(fill)
                    self.fill_history.append(fill)
                    
                    # Update order status
                    order.filled_quantity += fill.quantity
                    if order.filled_quantity >= order.quantity:
                        order.status = OrderStatus.FILLED
                        completed_orders.append(order_id)
                    else:
                        order.status = OrderStatus.PARTIAL_FILLED
        
        # Remove completed orders
        for order_id in completed_orders:
            completed_order = self.pending_orders.pop(order_id)
            self.order_history.append(completed_order)
        
        return fills
    
    def _should_fill_order(self, order: Order, current_price: float) -> bool:
        """Determine if an order should be filled"""
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:
                return current_price >= order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop triggered, now check if limit can be filled
            if order.side == OrderSide.BUY:
                if current_price >= order.stop_price:
                    return current_price <= order.price
            else:
                if current_price <= order.stop_price:
                    return current_price >= order.price
        
        return False
    
    def _execute_order(self, order: Order, current_price: float) -> Optional[Fill]:
        """Execute an order and return fill details"""
        # Calculate execution price with spread and slippage
        execution_price = self._calculate_execution_price(
            order, current_price
        )
        
        # Calculate fill quantity (may be partial)
        fill_quantity = self._calculate_fill_quantity(order, current_price)
        
        if fill_quantity <= 0:
            return None
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(
            order.symbol, fill_quantity, current_price
        )
        
        # Adjust execution price for market impact
        if order.side == OrderSide.BUY:
            execution_price += market_impact
        else:
            execution_price -= market_impact
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)
        
        fill = Fill(
            fill_id=f"FILL_{len(self.fill_history)}_{int(datetime.now().timestamp())}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            timestamp=datetime.now(),
            commission=commission,
            market_impact=market_impact
        )
        
        return fill
    
    def _calculate_execution_price(self, order: Order, current_price: float) -> float:
        """Calculate execution price including spread and slippage"""
        # Get base spread for asset class
        asset_class = self._get_asset_class(order.symbol)
        base_spread_bps = self.base_spread_bps.get(asset_class, 10)
        
        # Apply spread multiplier for market conditions
        effective_spread_bps = base_spread_bps * self.current_conditions.spread_multiplier
        spread = current_price * effective_spread_bps / 10000.0
        
        # Calculate bid/ask prices
        if order.side == OrderSide.BUY:
            # Buying at ask price
            execution_price = current_price + spread / 2
        else:
            # Selling at bid price
            execution_price = current_price - spread / 2
        
        # Add slippage for market orders and large sizes
        if order.order_type == OrderType.MARKET:
            slippage_factor = self._calculate_slippage_factor(order, current_price)
            slippage = current_price * slippage_factor
            
            if order.side == OrderSide.BUY:
                execution_price += slippage
            else:
                execution_price -= slippage
        
        return execution_price
    
    def _calculate_slippage_factor(self, order: Order, current_price: float) -> float:
        """Calculate slippage factor based on order size and market conditions"""
        # Base slippage from volatility and volume conditions
        base_slippage = 0.0001  # 1 bps base slippage
        
        # Volatility adjustment
        vol_adjustment = self.current_conditions.volatility * 0.5
        
        # Volume/liquidity adjustment
        liquidity_adjustment = (2.0 - self.current_conditions.liquidity_factor) * 0.5
        
        # Order size adjustment (simplified - would normally need average daily volume)
        notional_value = order.quantity * current_price
        size_adjustment = min(notional_value / 100000.0, 2.0)  # Cap at 2x for very large orders
        
        total_slippage = base_slippage * (1 + vol_adjustment + liquidity_adjustment + size_adjustment)
        
        return total_slippage
    
    def _calculate_fill_quantity(self, order: Order, current_price: float) -> float:
        """Calculate how much of the order gets filled (may be partial)"""
        remaining_quantity = order.quantity - order.filled_quantity
        
        if not self.liquidity_simulation:
            return remaining_quantity
        
        # Simulate partial fills based on market conditions
        fill_probability = min(self.current_conditions.liquidity_factor * 
                             self.current_conditions.volume_profile, 1.0)
        
        if random.random() > fill_probability:
            # Partial fill
            fill_ratio = random.uniform(0.3, 0.8)  # Fill 30-80% of remaining
            return remaining_quantity * fill_ratio
        else:
            # Full fill
            return remaining_quantity
    
    def _calculate_market_impact(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate market impact of the trade"""
        # Simplified market impact model: impact proportional to sqrt of order size
        notional_value = quantity * price
        impact_factor = self.market_impact_coefficient * self.current_conditions.market_impact_factor
        
        # Square root law for market impact
        market_impact = price * impact_factor * np.sqrt(notional_value / 100000.0) / 10000.0
        
        return market_impact
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        notional_value = quantity * price
        commission = max(notional_value * self.base_commission, self.min_commission)
        return commission
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol (simplified)"""
        if symbol.endswith('=X'):
            return 'forex'
        elif symbol.endswith('-USD'):
            return 'crypto'
        elif symbol in ['GLD', 'SLV', 'USO', 'UNG']:
            return 'commodities'
        elif symbol in ['TLT', 'IEF', 'SHY', 'LQD', 'HYG']:
            return 'bonds'
        else:
            return 'equity'
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self.order_history.append(order)
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        
        # Search in history
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        return None
    
    def get_trading_summary(self) -> Dict:
        """Get summary of trading activity"""
        total_trades = len(self.fill_history)
        total_commission = sum(fill.commission for fill in self.fill_history)
        total_market_impact = sum(fill.market_impact for fill in self.fill_history)
        
        return {
            'total_orders': len(self.order_history) + len(self.pending_orders),
            'pending_orders': len(self.pending_orders),
            'total_fills': total_trades,
            'total_commission': total_commission,
            'total_market_impact': total_market_impact,
            'avg_latency_ms': self.current_conditions.latency_ms,
            'current_spread_multiplier': self.current_conditions.spread_multiplier,
            'current_liquidity_factor': self.current_conditions.liquidity_factor
        }
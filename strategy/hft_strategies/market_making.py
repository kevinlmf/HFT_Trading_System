"""
Market Making Strategy for HFT
Posts limit orders on both sides of the book to capture bid-ask spread
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class InventoryManagement(Enum):
    """Inventory management modes"""
    NAIVE = "naive"  # No inventory management
    LINEAR = "linear"  # Linear skewing based on inventory
    EXPONENTIAL = "exponential"  # Exponential skewing


@dataclass
class MarketMakingParams:
    """Market making strategy parameters"""
    spread_multiplier: float = 1.5  # Multiplier of minimum tick
    order_size: float = 100.0  # Size per side
    max_inventory: int = 500  # Maximum absolute inventory
    skew_intensity: float = 0.5  # Inventory skew strength (0-1)
    inventory_mode: InventoryManagement = InventoryManagement.LINEAR
    tick_size: float = 0.01  # Minimum price increment
    adverse_selection_threshold: float = 0.02  # Stop quoting if price moves > this %


class MarketMakingStrategy:
    """
    Market making strategy that provides liquidity on both sides

    Key features:
    - Bid-ask spread capture
    - Inventory risk management via price skewing
    - Adverse selection protection
    - Position limits
    """

    def __init__(self, params: Optional[MarketMakingParams] = None):
        self.params = params or MarketMakingParams()
        self.inventory = 0
        self.last_mid_price = None
        self.total_pnl = 0.0
        self.quote_count = 0
        self.fill_count = 0

    def reset(self):
        """Reset strategy state"""
        self.inventory = 0
        self.last_mid_price = None
        self.total_pnl = 0.0
        self.quote_count = 0
        self.fill_count = 0

    def calculate_quotes(
        self,
        mid_price: float,
        volatility: Optional[float] = None,
        market_imbalance: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate bid and ask quote prices

        Args:
            mid_price: Current mid price
            volatility: Recent volatility (optional, for dynamic spreads)
            market_imbalance: Order book imbalance (optional, for asymmetric quoting)

        Returns:
            (bid_price, ask_price) or (None, None) if should not quote
        """
        # Check if we should stop quoting due to adverse selection
        if self.last_mid_price is not None:
            price_change = abs(mid_price - self.last_mid_price) / self.last_mid_price
            if price_change > self.params.adverse_selection_threshold:
                return None, None

        self.last_mid_price = mid_price

        # Check inventory limits
        if abs(self.inventory) >= self.params.max_inventory:
            # Only quote on the side that reduces inventory
            if self.inventory > 0:
                # Only offer (sell)
                ask_price = self._calculate_ask_price(mid_price, volatility)
                return None, ask_price
            else:
                # Only bid (buy)
                bid_price = self._calculate_bid_price(mid_price, volatility)
                return bid_price, None

        # Calculate base spread
        base_spread = self.params.tick_size * self.params.spread_multiplier

        # Adjust spread based on volatility
        if volatility is not None and volatility > 0:
            # Widen spread in high volatility
            vol_adjustment = 1.0 + (volatility * 10.0)  # Scale volatility impact
            base_spread *= vol_adjustment

        # Calculate inventory skew
        skew = self._calculate_inventory_skew()

        # Apply market imbalance
        imbalance_skew = 0.0
        if market_imbalance is not None:
            # If more buyers (positive imbalance), widen ask and tighten bid
            imbalance_skew = market_imbalance * self.params.tick_size

        # Calculate final prices
        bid_price = mid_price - base_spread / 2 - skew - imbalance_skew
        ask_price = mid_price + base_spread / 2 - skew + imbalance_skew

        # Round to tick size
        bid_price = self._round_to_tick(bid_price)
        ask_price = self._round_to_tick(ask_price)

        self.quote_count += 1

        return bid_price, ask_price

    def _calculate_inventory_skew(self) -> float:
        """
        Calculate price skew based on current inventory
        Positive skew = shift quotes upward (encourage selling)
        Negative skew = shift quotes downward (encourage buying)
        """
        if self.inventory == 0:
            return 0.0

        inventory_ratio = self.inventory / self.params.max_inventory

        if self.params.inventory_mode == InventoryManagement.NAIVE:
            return 0.0

        elif self.params.inventory_mode == InventoryManagement.LINEAR:
            # Linear skewing
            skew = inventory_ratio * self.params.skew_intensity * self.params.tick_size * 5
            return skew

        elif self.params.inventory_mode == InventoryManagement.EXPONENTIAL:
            # Exponential skewing (more aggressive at extremes)
            sign = np.sign(inventory_ratio)
            skew = sign * (np.exp(abs(inventory_ratio) * 2) - 1) * self.params.skew_intensity * self.params.tick_size * 5
            return skew

        return 0.0

    def _calculate_bid_price(self, mid_price: float, volatility: Optional[float]) -> float:
        """Calculate bid price when only quoting bid side"""
        base_spread = self.params.tick_size * self.params.spread_multiplier
        if volatility is not None and volatility > 0:
            base_spread *= (1.0 + volatility * 10.0)

        skew = self._calculate_inventory_skew()
        bid_price = mid_price - base_spread / 2 - skew
        return self._round_to_tick(bid_price)

    def _calculate_ask_price(self, mid_price: float, volatility: Optional[float]) -> float:
        """Calculate ask price when only quoting ask side"""
        base_spread = self.params.tick_size * self.params.spread_multiplier
        if volatility is not None and volatility > 0:
            base_spread *= (1.0 + volatility * 10.0)

        skew = self._calculate_inventory_skew()
        ask_price = mid_price + base_spread / 2 - skew
        return self._round_to_tick(ask_price)

    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size"""
        return round(price / self.params.tick_size) * self.params.tick_size

    def on_fill(self, side: str, price: float, quantity: float):
        """
        Update state when order is filled

        Args:
            side: 'buy' or 'sell'
            price: Fill price
            quantity: Fill quantity
        """
        self.fill_count += 1

        if side.lower() == 'buy':
            self.inventory += quantity
            self.total_pnl -= price * quantity  # Cash outflow
        elif side.lower() == 'sell':
            self.inventory -= quantity
            self.total_pnl += price * quantity  # Cash inflow

    def get_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics"""
        fill_rate = self.fill_count / self.quote_count if self.quote_count > 0 else 0.0

        return {
            'inventory': self.inventory,
            'total_pnl': self.total_pnl,
            'quote_count': self.quote_count,
            'fill_count': self.fill_count,
            'fill_rate': fill_rate,
            'avg_pnl_per_fill': self.total_pnl / self.fill_count if self.fill_count > 0 else 0.0
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for backtesting framework compatibility

        Returns:
            Series with values: -1 (sell), 0 (hold), 1 (buy)
        """
        signals = pd.Series(0, index=data.index)

        if 'close' not in data.columns:
            return signals

        # Calculate rolling volatility
        if len(data) < 20:
            return signals

        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()

        for i in range(20, len(data)):
            mid_price = data['close'].iloc[i]
            vol = volatility.iloc[i] if pd.notna(volatility.iloc[i]) else None

            bid, ask = self.calculate_quotes(mid_price, vol)

            if bid is None and ask is None:
                signals.iloc[i] = 0
            elif bid is None:
                # Only selling
                signals.iloc[i] = -1
            elif ask is None:
                # Only buying
                signals.iloc[i] = 1
            else:
                # Market making both sides - hold or adjust based on inventory
                if self.inventory > 0:
                    signals.iloc[i] = -1  # Prefer selling to reduce inventory
                elif self.inventory < 0:
                    signals.iloc[i] = 1  # Prefer buying to reduce inventory
                else:
                    signals.iloc[i] = 0

        return signals


def calculate_order_book_imbalance(bid_volume: float, ask_volume: float) -> float:
    """
    Calculate order book imbalance

    Returns:
        Value in [-1, 1] where:
        - Positive = more demand (buying pressure)
        - Negative = more supply (selling pressure)
    """
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    return (bid_volume - ask_volume) / total


# Example usage
if __name__ == "__main__":
    # Create strategy with custom parameters
    params = MarketMakingParams(
        spread_multiplier=2.0,
        order_size=100.0,
        max_inventory=1000,
        skew_intensity=0.7,
        inventory_mode=InventoryManagement.LINEAR,
        tick_size=0.01
    )

    strategy = MarketMakingStrategy(params)

    # Simulate market making
    mid_price = 100.0
    volatility = 0.02

    for i in range(10):
        bid, ask = strategy.calculate_quotes(mid_price, volatility)
        print(f"Step {i}: Bid={bid:.2f}, Ask={ask:.2f}, Inventory={strategy.inventory}")

        # Simulate random fills
        if np.random.random() < 0.3:
            side = 'buy' if np.random.random() < 0.5 else 'sell'
            price = bid if side == 'buy' else ask
            strategy.on_fill(side, price, 10.0)

        # Random price walk
        mid_price += np.random.normal(0, 0.05)

    print("\nFinal Metrics:")
    for key, value in strategy.get_metrics().items():
        print(f"{key}: {value:.4f}")

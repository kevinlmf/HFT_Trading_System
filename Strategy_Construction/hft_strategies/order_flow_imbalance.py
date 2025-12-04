"""
Order Flow Imbalance Strategy for HFT
Exploits temporary imbalances in order book to predict short-term price movements
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque


@dataclass
class OrderFlowParams:
    """Order flow imbalance parameters"""
    lookback_periods: int = 10  # Number of periods to calculate imbalance
    imbalance_threshold: float = 0.3  # Threshold for significant imbalance (0-1)
    depth_levels: int = 5  # Number of order book levels to consider
    volume_threshold: float = 1000.0  # Minimum volume to generate signal
    hold_periods: int = 5  # Number of periods to hold position
    decay_factor: float = 0.9  # Exponential decay for historical imbalances


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time"""
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]  # [(price, volume), ...]
    mid_price: float
    spread: float


class OrderFlowImbalanceStrategy:
    """
    Order flow imbalance strategy for high-frequency trading

    Key concepts:
    - Order book imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    - Positive imbalance suggests buying pressure -> price likely to increase
    - Negative imbalance suggests selling pressure -> price likely to decrease

    Features:
    - Multi-level order book analysis
    - Exponentially weighted imbalance
    - Volume-weighted signals
    - Trade execution timing
    """

    def __init__(self, params: Optional[OrderFlowParams] = None):
        self.params = params or OrderFlowParams()
        self.imbalance_history: deque = deque(maxlen=self.params.lookback_periods)
        self.position = 0
        self.holding_timer = 0
        self.total_trades = 0
        self.successful_trades = 0

    def reset(self):
        """Reset strategy state"""
        self.imbalance_history.clear()
        self.position = 0
        self.holding_timer = 0
        self.total_trades = 0
        self.successful_trades = 0

    def calculate_order_book_imbalance(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        depth_levels: Optional[int] = None
    ) -> float:
        """
        Calculate order book imbalance

        Args:
            bids: List of (price, volume) for bid side
            asks: List of (price, volume) for ask side
            depth_levels: Number of levels to consider (uses params if None)

        Returns:
            Imbalance in [-1, 1] where:
            - +1 = strong buying pressure
            - -1 = strong selling pressure
            - 0 = balanced
        """
        depth = depth_levels or self.params.depth_levels

        # Sum volumes up to depth levels
        bid_volume = sum(vol for _, vol in bids[:depth])
        ask_volume = sum(vol for _, vol in asks[:depth])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / total_volume
        return imbalance

    def calculate_weighted_imbalance(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        mid_price: float
    ) -> float:
        """
        Calculate volume-weighted order book imbalance

        Gives more weight to orders closer to mid price
        """
        weighted_bid_volume = 0.0
        weighted_ask_volume = 0.0

        for i, (price, volume) in enumerate(bids[:self.params.depth_levels]):
            # Weight by proximity to mid price
            distance = abs(price - mid_price) / mid_price
            weight = np.exp(-distance * 100)  # Exponential decay
            weighted_bid_volume += volume * weight

        for i, (price, volume) in enumerate(asks[:self.params.depth_levels]):
            distance = abs(price - mid_price) / mid_price
            weight = np.exp(-distance * 100)
            weighted_ask_volume += volume * weight

        total = weighted_bid_volume + weighted_ask_volume
        if total == 0:
            return 0.0

        return (weighted_bid_volume - weighted_ask_volume) / total

    def calculate_exponential_imbalance(self) -> float:
        """
        Calculate exponentially weighted average of historical imbalances

        Recent imbalances have higher weight
        """
        if len(self.imbalance_history) == 0:
            return 0.0

        weights = np.array([
            self.params.decay_factor ** i
            for i in range(len(self.imbalance_history) - 1, -1, -1)
        ])

        imbalances = np.array(list(self.imbalance_history))
        weighted_imbalance = np.average(imbalances, weights=weights)

        return weighted_imbalance

    def generate_signal(
        self,
        snapshot: OrderBookSnapshot,
        trade_volume: Optional[float] = None
    ) -> int:
        """
        Generate trading signal based on order flow imbalance

        Args:
            snapshot: Current order book snapshot
            trade_volume: Recent trade volume (optional)

        Returns:
            Signal: -1 (sell), 0 (hold), 1 (buy)
        """
        # Calculate current imbalance
        imbalance = self.calculate_weighted_imbalance(
            snapshot.bids,
            snapshot.asks,
            snapshot.mid_price
        )

        # Add to history
        self.imbalance_history.append(imbalance)

        # Manage existing position
        if self.position != 0:
            self.holding_timer += 1

            # Exit condition: holding period expired
            if self.holding_timer >= self.params.hold_periods:
                exit_signal = -self.position  # Opposite of current position
                self.position = 0
                self.holding_timer = 0
                return exit_signal

            # Early exit: imbalance reversed significantly
            if (self.position > 0 and imbalance < -self.params.imbalance_threshold) or \
               (self.position < 0 and imbalance > self.params.imbalance_threshold):
                exit_signal = -self.position
                self.position = 0
                self.holding_timer = 0
                return exit_signal

            return 0  # Hold current position

        # Check volume threshold
        if trade_volume is not None and trade_volume < self.params.volume_threshold:
            return 0

        # Calculate exponentially weighted imbalance
        ema_imbalance = self.calculate_exponential_imbalance()

        # Generate entry signal
        if ema_imbalance > self.params.imbalance_threshold:
            # Strong buying pressure -> buy
            self.position = 1
            self.holding_timer = 0
            self.total_trades += 1
            return 1

        elif ema_imbalance < -self.params.imbalance_threshold:
            # Strong selling pressure -> sell
            self.position = -1
            self.holding_timer = 0
            self.total_trades += 1
            return -1

        return 0

    def calculate_price_impact(
        self,
        order_size: float,
        side: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> float:
        """
        Estimate price impact of an order

        Args:
            order_size: Size of order
            side: 'buy' or 'sell'
            bids: Order book bids
            asks: Order book asks

        Returns:
            Estimated price impact (percentage)
        """
        levels = asks if side == 'buy' else bids
        remaining = order_size
        total_cost = 0.0

        for price, volume in levels:
            if remaining <= 0:
                break

            fill_volume = min(remaining, volume)
            total_cost += price * fill_volume
            remaining -= fill_volume

        if remaining > 0:
            # Not enough liquidity
            return float('inf')

        avg_price = total_cost / order_size
        reference_price = levels[0][0]  # Best price

        impact = abs(avg_price - reference_price) / reference_price
        return impact

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price/volume data

        For compatibility with backtesting framework.
        Note: Real HFT implementation would use order book data.
        """
        signals = pd.Series(0, index=data.index)

        if 'close' not in data.columns or 'volume' not in data.columns:
            return signals

        # Simulate order book from OHLCV data
        for i in range(self.params.lookback_periods, len(data)):
            close = data['close'].iloc[i]
            volume = data['volume'].iloc[i]

            # Simplified: use close and volume to simulate order book imbalance
            # In real HFT, would use actual order book data

            # Calculate returns and volume change
            if i > 0:
                price_change = (close - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                volume_ratio = volume / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1.0

                # Approximate imbalance from price momentum and volume
                approx_imbalance = np.tanh(price_change * 100) * min(volume_ratio, 2.0) / 2.0
                self.imbalance_history.append(approx_imbalance)

            # Create synthetic order book snapshot
            spread = close * 0.0001  # 1 bps spread
            bids = [(close - spread/2, volume/2)]
            asks = [(close + spread/2, volume/2)]

            snapshot = OrderBookSnapshot(
                timestamp=data.index[i],
                bids=bids,
                asks=asks,
                mid_price=close,
                spread=spread
            )

            # Generate signal
            signal = self.generate_signal(snapshot, volume)
            signals.iloc[i] = signal

        return signals

    def get_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics"""
        success_rate = (
            self.successful_trades / self.total_trades
            if self.total_trades > 0 else 0.0
        )

        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': success_rate,
            'current_position': self.position,
            'avg_imbalance': np.mean(list(self.imbalance_history)) if self.imbalance_history else 0.0
        }


# Example usage
if __name__ == "__main__":
    # Create strategy
    params = OrderFlowParams(
        lookback_periods=20,
        imbalance_threshold=0.3,
        depth_levels=5,
        hold_periods=3
    )

    strategy = OrderFlowImbalanceStrategy(params)

    # Simulate order book snapshots
    np.random.seed(42)
    mid_price = 100.0

    for i in range(50):
        # Generate random order book
        bid_volumes = np.random.exponential(1000, 5)
        ask_volumes = np.random.exponential(1000, 5)

        # Add systematic imbalance
        if i % 10 < 5:
            bid_volumes *= 1.5  # Buying pressure
        else:
            ask_volumes *= 1.5  # Selling pressure

        bids = [(mid_price - 0.01 * (j+1), vol) for j, vol in enumerate(bid_volumes)]
        asks = [(mid_price + 0.01 * (j+1), vol) for j, vol in enumerate(ask_volumes)]

        snapshot = OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread=0.02
        )

        # Generate signal
        signal = strategy.generate_signal(snapshot, sum(bid_volumes) + sum(ask_volumes))

        if signal != 0:
            print(f"Step {i}: Signal={signal}, Position={strategy.position}")

        # Random price walk
        mid_price += np.random.normal(0, 0.05)

    print("\nFinal Metrics:")
    for key, value in strategy.get_metrics().items():
        print(f"{key}: {value:.4f}")

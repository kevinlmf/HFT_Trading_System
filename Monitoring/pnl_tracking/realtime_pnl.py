"""
Real-Time PnL Tracker

Tracks profit & loss in real-time with:
- Position tracking
- Mark-to-market P&L
- Realized/unrealized P&L
- Cost basis tracking
- Performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Current position in a symbol"""
    symbol: str
    quantity: float
    avg_entry_price: float
    entry_time: datetime
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, price: float):
        """Update mark-to-market price"""
        self.last_price = price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity

    def get_market_value(self) -> float:
        """Get current market value"""
        return self.last_price * abs(self.quantity)

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    strategy: str = ""
    pnl: float = 0.0  # Filled after close

    def get_notional(self) -> float:
        """Get trade notional value"""
        return self.quantity * self.price


class RealTimePnLTracker:
    """
    Real-time P&L tracking system

    Features:
    - Track positions across multiple symbols
    - Calculate realized and unrealized P&L
    - Performance metrics (Sharpe, drawdown, win rate)
    - Risk metrics (exposure, concentration)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0001,  # 0.01%
        max_position_size: float = 0.2,  # 20% of capital per position
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # P&L tracking
        self.daily_pnl = []
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0

        # High water mark for drawdown
        self.high_water_mark = initial_capital

    def update_market_price(self, symbol: str, price: float):
        """Update mark-to-market price for a symbol"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)

    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str = "",
        timestamp: Optional[datetime] = None
    ) -> Optional[Trade]:
        """
        Open or add to a position

        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive for long, negative for short)
            price: Execution price
            strategy: Strategy name
            timestamp: Trade timestamp

        Returns:
            Trade object if successful, None if rejected
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate commission
        commission = abs(quantity * price) * self.commission_rate
        self.total_commission += commission

        # Check position size limit
        notional = abs(quantity * price)
        if notional > self.current_capital * self.max_position_size:
            logger.warning(f"Position size too large for {symbol}: ${notional:.2f}")
            return None

        # Create trade record
        trade = Trade(
            trade_id=f"{symbol}_{len(self.trades)}",
            timestamp=timestamp,
            symbol=symbol,
            side='buy' if quantity > 0 else 'sell',
            quantity=abs(quantity),
            price=price,
            commission=commission,
            strategy=strategy
        )
        self.trades.append(trade)
        self.total_trades += 1

        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Calculate new average price
            total_cost = pos.avg_entry_price * pos.quantity + price * quantity
            new_quantity = pos.quantity + quantity

            if abs(new_quantity) < 1e-6:  # Position closed
                # Realize P&L
                pos.realized_pnl += (price - pos.avg_entry_price) * abs(quantity)
                trade.pnl = pos.realized_pnl
                logger.info(f"Closed position {symbol}: P&L = ${pos.realized_pnl:.2f}")
                del self.positions[symbol]
            else:
                pos.quantity = new_quantity
                pos.avg_entry_price = total_cost / new_quantity
                pos.last_price = price
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                entry_time=timestamp,
                last_price=price
            )
            logger.info(f"Opened position {symbol}: {quantity} @ ${price:.2f}")

        # Update capital
        self.current_capital -= commission

        return trade

    def close_position(self, symbol: str, price: float, strategy: str = "") -> Optional[float]:
        """
        Close entire position in a symbol

        Returns:
            Realized P&L if successful, None if no position
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self.positions[symbol]
        # Close with opposite quantity
        close_quantity = -pos.quantity

        trade = self.open_position(symbol, close_quantity, price, strategy)

        if trade:
            return trade.pnl
        return None

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        total = 0.0

        # Unrealized P&L from open positions
        for pos in self.positions.values():
            total += pos.unrealized_pnl

        # Realized P&L from closed positions
        for trade in self.trades:
            total += trade.pnl

        return total - self.total_commission

    def get_equity(self) -> float:
        """Get current equity (capital + P&L)"""
        return self.initial_capital + self.get_total_pnl()

    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of current positions"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Quantity': pos.quantity,
                'Entry Price': f"${pos.avg_entry_price:.2f}",
                'Current Price': f"${pos.last_price:.2f}",
                'Unrealized P&L': f"${pos.unrealized_pnl:.2f}",
                'Market Value': f"${pos.get_market_value():.2f}",
                'P&L %': f"{(pos.unrealized_pnl / (pos.avg_entry_price * abs(pos.quantity))) * 100:.2f}%"
            })

        return pd.DataFrame(data)

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        total_pnl = self.get_total_pnl()
        equity = self.get_equity()

        # Win rate
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0

        # Return
        total_return = (equity - self.initial_capital) / self.initial_capital

        # Sharpe ratio (from daily P&L)
        if len(self.daily_pnl) > 1:
            returns = np.array(self.daily_pnl)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equity_curve = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Exposure
        total_exposure = sum(pos.get_market_value() for pos in self.positions.values())
        exposure_ratio = total_exposure / equity if equity > 0 else 0

        return {
            'Total P&L': f"${total_pnl:.2f}",
            'Total Return': f"{total_return * 100:.2f}%",
            'Current Equity': f"${equity:.2f}",
            'Total Trades': self.total_trades,
            'Win Rate': f"{win_rate * 100:.1f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown * 100:.2f}%",
            'Total Commission': f"${self.total_commission:.2f}",
            'Exposure': f"{exposure_ratio * 100:.1f}%",
            'Open Positions': len(self.positions)
        }

    def print_summary(self):
        """Print P&L summary to console"""
        print("\n" + "=" * 70)
        print("REAL-TIME P&L SUMMARY")
        print("=" * 70)

        # Performance metrics
        metrics = self.get_performance_metrics()
        for key, value in metrics.items():
            print(f"{key:.<40} {value:>25}")

        # Open positions
        if self.positions:
            print("\n" + "-" * 70)
            print("OPEN POSITIONS")
            print("-" * 70)
            print(self.get_positions_summary().to_string(index=False))

        print("=" * 70 + "\n")

    def snapshot(self, timestamp: Optional[datetime] = None):
        """Take a snapshot of current equity for tracking"""
        if timestamp is None:
            timestamp = datetime.now()

        equity = self.get_equity()
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)

        # Update high water mark
        if equity > self.high_water_mark:
            self.high_water_mark = equity

        # Calculate daily P&L
        if len(self.equity_curve) > 1:
            daily_return = (equity - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_pnl.append(daily_return)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'Timestamp': self.timestamps,
            'Equity': self.equity_curve
        })


# Example usage
if __name__ == "__main__":
    tracker = RealTimePnLTracker(initial_capital=100000)

    # Simulate some trades
    tracker.open_position('AAPL', 100, 150.0, strategy='momentum')
    tracker.update_market_price('AAPL', 152.0)

    tracker.open_position('MSFT', 50, 300.0, strategy='pairs')
    tracker.update_market_price('MSFT', 305.0)

    tracker.snapshot()

    # Print summary
    tracker.print_summary()

"""
P&L Analysis and Attribution for Trading Strategies
Breaks down profit/loss into components: trading, costs, slippage, etc.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PnLComponent(Enum):
    """P&L components for attribution"""
    GROSS_PNL = "gross_pnl"  # Before costs
    TRANSACTION_COSTS = "transaction_costs"  # Commissions, fees
    SLIPPAGE = "slippage"  # Execution slippage
    MARKET_IMPACT = "market_impact"  # Price impact
    FINANCING_COSTS = "financing_costs"  # Borrow costs, interest
    NET_PNL = "net_pnl"  # Final P&L


@dataclass
class PnLBreakdown:
    """Detailed P&L breakdown"""
    gross_pnl: float
    transaction_costs: float
    slippage: float
    market_impact: float
    financing_costs: float
    net_pnl: float

    # Time-series data
    gross_pnl_series: Optional[pd.Series] = None
    net_pnl_series: Optional[pd.Series] = None
    cost_series: Optional[pd.Series] = None

    # Attribution by source
    pnl_by_strategy: Optional[Dict[str, float]] = None
    pnl_by_asset: Optional[Dict[str, float]] = None
    pnl_by_time: Optional[pd.DataFrame] = None

    # Statistics
    avg_trade_pnl: float = 0.0
    median_trade_pnl: float = 0.0
    pnl_std: float = 0.0
    pnl_sharpe: float = 0.0


class PnLAnalyzer:
    """
    Comprehensive P&L analyzer with cost attribution

    Features:
    - Gross vs net P&L calculation
    - Transaction cost breakdown
    - Slippage analysis
    - Market impact estimation
    - Time-based attribution
    - Strategy/asset attribution
    """

    def __init__(
        self,
        commission_rate: float = 0.0001,  # 1 bp
        spread_cost: float = 0.0001,  # 1 bp
        market_impact_coef: float = 0.1,  # Price impact coefficient
        financing_rate: float = 0.02  # 2% annual
    ):
        self.commission_rate = commission_rate
        self.spread_cost = spread_cost
        self.market_impact_coef = market_impact_coef
        self.financing_rate = financing_rate

    def calculate_transaction_costs(
        self,
        trades: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate transaction costs from trades

        Args:
            trades: DataFrame with columns ['price', 'quantity', 'side']

        Returns:
            Series of transaction costs per trade
        """
        notional = trades['price'] * trades['quantity'].abs()
        costs = notional * self.commission_rate
        return costs

    def calculate_slippage(
        self,
        trades: pd.DataFrame,
        mid_prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate slippage costs

        Args:
            trades: DataFrame with columns ['execution_price', 'quantity', 'side']
            mid_prices: Series of mid prices at execution time (optional)

        Returns:
            Series of slippage costs
        """
        if mid_prices is None:
            # Estimate slippage from spread
            notional = trades['price'] * trades['quantity'].abs()
            slippage = notional * self.spread_cost / 2
        else:
            # Calculate actual slippage
            execution_prices = trades['price']
            trade_indices = trades.index

            slippage = []
            for idx, row in trades.iterrows():
                if idx in mid_prices.index:
                    mid = mid_prices.loc[idx]
                    exec_price = row['price']
                    quantity = abs(row['quantity'])

                    # Slippage is difference between execution and mid
                    slip = abs(exec_price - mid) * quantity
                    slippage.append(slip)
                else:
                    # Fallback to spread estimate
                    slippage.append(row['price'] * abs(row['quantity']) * self.spread_cost / 2)

            slippage = pd.Series(slippage, index=trade_indices)

        return slippage

    def calculate_market_impact(
        self,
        trades: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Estimate market impact costs

        Args:
            trades: DataFrame with columns ['price', 'quantity', 'side']
            volume_data: DataFrame with average daily volume per asset

        Returns:
            Series of market impact costs
        """
        market_impact = []

        for idx, row in trades.iterrows():
            quantity = abs(row['quantity'])
            price = row['price']

            if volume_data is not None and 'symbol' in row:
                symbol = row['symbol']
                if symbol in volume_data.index:
                    avg_volume = volume_data.loc[symbol, 'avg_volume']
                    # Impact proportional to trade size / average volume
                    participation_rate = quantity / avg_volume if avg_volume > 0 else 0.1
                else:
                    participation_rate = 0.1
            else:
                # Default participation rate
                participation_rate = 0.1

            # Square-root model for market impact
            impact = price * quantity * self.market_impact_coef * np.sqrt(participation_rate)
            market_impact.append(impact)

        return pd.Series(market_impact, index=trades.index)

    def calculate_financing_costs(
        self,
        positions: pd.DataFrame,
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate financing costs for positions

        Args:
            positions: DataFrame with columns ['value', 'duration'] (duration in days)
            periods_per_year: Trading periods per year

        Returns:
            Series of financing costs
        """
        daily_rate = self.financing_rate / periods_per_year

        if 'duration' in positions.columns:
            costs = positions['value'].abs() * daily_rate * positions['duration']
        else:
            # Single period
            costs = positions['value'].abs() * daily_rate

        return costs

    def analyze_pnl(
        self,
        trades: pd.DataFrame,
        positions: Optional[pd.DataFrame] = None,
        mid_prices: Optional[pd.Series] = None,
        volume_data: Optional[pd.DataFrame] = None
    ) -> PnLBreakdown:
        """
        Comprehensive P&L analysis

        Args:
            trades: DataFrame with trade history
            positions: DataFrame with position history (optional)
            mid_prices: Series of mid prices (optional)
            volume_data: Volume data for impact estimation (optional)

        Returns:
            PnLBreakdown with full attribution
        """
        # Calculate P&L components
        if 'pnl' in trades.columns:
            gross_pnl = trades['pnl'].sum()
            gross_pnl_series = trades['pnl'].cumsum()
        else:
            # Calculate from trades
            gross_pnl = 0.0
            gross_pnl_series = pd.Series(0.0, index=trades.index)

        # Transaction costs
        transaction_costs = self.calculate_transaction_costs(trades).sum()

        # Slippage
        slippage = self.calculate_slippage(trades, mid_prices).sum()

        # Market impact
        market_impact = self.calculate_market_impact(trades, volume_data).sum()

        # Financing costs
        if positions is not None:
            financing_costs = self.calculate_financing_costs(positions).sum()
        else:
            financing_costs = 0.0

        # Net P&L
        net_pnl = gross_pnl - transaction_costs - slippage - market_impact - financing_costs

        # Calculate net P&L series
        total_costs = (
            self.calculate_transaction_costs(trades) +
            self.calculate_slippage(trades, mid_prices) +
            self.calculate_market_impact(trades, volume_data)
        )
        cost_series = total_costs.cumsum()

        if 'pnl' in trades.columns:
            net_pnl_series = gross_pnl_series - cost_series
        else:
            net_pnl_series = -cost_series

        # Attribution analysis
        pnl_by_asset = None
        if 'symbol' in trades.columns and 'pnl' in trades.columns:
            pnl_by_asset = trades.groupby('symbol')['pnl'].sum().to_dict()

        pnl_by_strategy = None
        if 'strategy' in trades.columns and 'pnl' in trades.columns:
            pnl_by_strategy = trades.groupby('strategy')['pnl'].sum().to_dict()

        # Time-based attribution
        pnl_by_time = None
        if 'pnl' in trades.columns and isinstance(trades.index, pd.DatetimeIndex):
            pnl_by_time = pd.DataFrame({
                'gross_pnl': trades['pnl'].resample('D').sum(),
                'costs': total_costs.resample('D').sum()
            })
            pnl_by_time['net_pnl'] = pnl_by_time['gross_pnl'] - pnl_by_time['costs']

        # Statistics
        if 'pnl' in trades.columns:
            avg_trade_pnl = trades['pnl'].mean()
            median_trade_pnl = trades['pnl'].median()
            pnl_std = trades['pnl'].std()
            pnl_sharpe = avg_trade_pnl / pnl_std if pnl_std > 0 else 0.0
        else:
            avg_trade_pnl = 0.0
            median_trade_pnl = 0.0
            pnl_std = 0.0
            pnl_sharpe = 0.0

        return PnLBreakdown(
            gross_pnl=gross_pnl,
            transaction_costs=transaction_costs,
            slippage=slippage,
            market_impact=market_impact,
            financing_costs=financing_costs,
            net_pnl=net_pnl,
            gross_pnl_series=gross_pnl_series,
            net_pnl_series=net_pnl_series,
            cost_series=cost_series,
            pnl_by_strategy=pnl_by_strategy,
            pnl_by_asset=pnl_by_asset,
            pnl_by_time=pnl_by_time,
            avg_trade_pnl=avg_trade_pnl,
            median_trade_pnl=median_trade_pnl,
            pnl_std=pnl_std,
            pnl_sharpe=pnl_sharpe
        )

    def print_pnl_report(self, breakdown: PnLBreakdown):
        """Print formatted P&L report"""
        print("=" * 70)
        print("P&L BREAKDOWN REPORT")
        print("=" * 70)
        print(f"\nGross P&L:              ${breakdown.gross_pnl:>15,.2f}")
        print(f"Transaction Costs:      ${breakdown.transaction_costs:>15,.2f}")
        print(f"Slippage:               ${breakdown.slippage:>15,.2f}")
        print(f"Market Impact:          ${breakdown.market_impact:>15,.2f}")
        print(f"Financing Costs:        ${breakdown.financing_costs:>15,.2f}")
        print("-" * 70)
        print(f"Net P&L:                ${breakdown.net_pnl:>15,.2f}")
        print("=" * 70)

        # Cost percentage
        total_costs = (
            breakdown.transaction_costs +
            breakdown.slippage +
            breakdown.market_impact +
            breakdown.financing_costs
        )

        if breakdown.gross_pnl != 0:
            cost_pct = (total_costs / abs(breakdown.gross_pnl)) * 100
            print(f"\nTotal Costs as % of Gross P&L: {cost_pct:.2f}%")

        # Trade statistics
        print(f"\nTrade Statistics:")
        print(f"  Average Trade P&L:    ${breakdown.avg_trade_pnl:>15,.2f}")
        print(f"  Median Trade P&L:     ${breakdown.median_trade_pnl:>15,.2f}")
        print(f"  P&L Std Dev:          ${breakdown.pnl_std:>15,.2f}")
        print(f"  P&L Sharpe:           {breakdown.pnl_sharpe:>15,.2f}")

        # Attribution
        if breakdown.pnl_by_asset:
            print(f"\nP&L by Asset:")
            for asset, pnl in sorted(breakdown.pnl_by_asset.items(), key=lambda x: x[1], reverse=True):
                print(f"  {asset:<20} ${pnl:>15,.2f}")

        if breakdown.pnl_by_strategy:
            print(f"\nP&L by Strategy:")
            for strategy, pnl in sorted(breakdown.pnl_by_strategy.items(), key=lambda x: x[1], reverse=True):
                print(f"  {strategy:<20} ${pnl:>15,.2f}")

        print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Generate sample trade data
    np.random.seed(42)
    n_trades = 100

    trades = pd.DataFrame({
        'price': 100 + np.random.randn(n_trades) * 5,
        'quantity': np.random.randint(10, 100, n_trades),
        'side': np.random.choice(['buy', 'sell'], n_trades),
        'pnl': np.random.randn(n_trades) * 100,
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], n_trades)
    })

    # Create analyzer
    analyzer = PnLAnalyzer(
        commission_rate=0.0001,
        spread_cost=0.0001,
        market_impact_coef=0.1
    )

    # Analyze P&L
    breakdown = analyzer.analyze_pnl(trades)

    # Print report
    analyzer.print_pnl_report(breakdown)

"""
Adaptive Strategy Router

Automatically selects the best strategy based on:
1. Market regime detection (trending, mean-reverting, volatile)
2. Recent strategy performance
3. Strategy confidence scores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Track individual strategy performance"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    last_signal_time: Optional[datetime] = None
    confidence_score: float = 0.5  # 0-1 scale

    def update(self, pnl: float, timestamp: datetime):
        """Update performance metrics"""
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        self.recent_returns.append(pnl)
        self.last_signal_time = timestamp

    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.5
        return self.winning_trades / self.total_trades

    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent returns"""
        if len(self.recent_returns) < 10:
            return 0.0
        returns = np.array(self.recent_returns)
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def get_score(self) -> float:
        """
        Calculate overall strategy score
        Combines: win rate, Sharpe ratio, recent performance
        """
        win_rate = self.get_win_rate()
        sharpe = max(0, self.get_sharpe_ratio())

        # Recent performance (last 20 trades)
        recent_pnl = sum(list(self.recent_returns)[-20:]) if self.recent_returns else 0

        # Weighted score
        score = (
            0.3 * win_rate +
            0.3 * min(sharpe / 3.0, 1.0) +  # Normalize Sharpe
            0.2 * (1 if recent_pnl > 0 else 0) +
            0.2 * self.confidence_score
        )

        return score


@dataclass
class MarketRegime:
    """Detected market regime"""
    regime_type: str  # 'trending', 'mean_reverting', 'volatile', 'quiet'
    confidence: float  # 0-1
    volatility: float
    trend_strength: float
    timestamp: datetime


class AdaptiveStrategyRouter:
    """
    Intelligently route to the best strategy based on:
    - Market conditions
    - Strategy performance
    - Risk constraints
    """

    def __init__(
        self,
        strategies: Dict[str, object],
        lookback_window: int = 100,
        regime_window: int = 50,
        min_trades_for_selection: int = 10
    ):
        """
        Args:
            strategies: Dict of {strategy_name: strategy_instance}
            lookback_window: Number of data points for performance tracking
            regime_window: Window for market regime detection
            min_trades_for_selection: Minimum trades before trusting strategy score
        """
        self.strategies = strategies
        self.lookback_window = lookback_window
        self.regime_window = regime_window
        self.min_trades_for_selection = min_trades_for_selection

        # Performance tracking
        self.performance: Dict[str, StrategyPerformance] = {
            name: StrategyPerformance(strategy_name=name)
            for name in strategies.keys()
        }

        # Market data buffer
        self.price_history = deque(maxlen=regime_window)
        self.volume_history = deque(maxlen=regime_window)

        # Current state
        self.current_regime: Optional[MarketRegime] = None
        self.active_strategy: Optional[str] = None

        # Strategy-regime mapping (which strategies work best in which regimes)
        self.regime_strategy_map = {
            'trending': ['momentum', 'lstm', 'transformer'],
            'mean_reverting': ['pairs_trading', 'stat_arb', 'mean_variance'],
            'volatile': ['market_making', 'volatility_arb'],
            'quiet': ['market_making', 'order_flow']
        }

    def detect_market_regime(self) -> MarketRegime:
        """
        Detect current market regime using technical indicators

        Returns:
            MarketRegime with type and confidence
        """
        if len(self.price_history) < self.regime_window:
            return MarketRegime(
                regime_type='unknown',
                confidence=0.0,
                volatility=0.0,
                trend_strength=0.0,
                timestamp=datetime.now()
            )

        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Trend strength (linear regression R²)
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        y_pred = slope * x + np.mean(prices)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Hurst exponent (mean reversion vs trending)
        hurst = self._calculate_hurst(prices)

        # Regime classification
        if hurst > 0.6 and r_squared > 0.5:
            regime_type = 'trending'
            confidence = min(r_squared, hurst - 0.5) * 2
        elif hurst < 0.4:
            regime_type = 'mean_reverting'
            confidence = (0.5 - hurst) * 2
        elif volatility > 0.3:
            regime_type = 'volatile'
            confidence = min(volatility / 0.5, 1.0)
        else:
            regime_type = 'quiet'
            confidence = 1 - volatility / 0.3

        return MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            volatility=volatility,
            trend_strength=r_squared,
            timestamp=datetime.now()
        )

    def _calculate_hurst(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent
        H > 0.5: trending
        H < 0.5: mean reverting
        H = 0.5: random walk
        """
        if len(prices) < 4:
            return 0.5
        
        lags = range(2, min(20, len(prices) // 2))
        tau = []
        valid_lags = []
        
        for lag in lags:
            if lag >= len(prices):
                continue
            std = np.sqrt(np.mean((prices[lag:] - prices[:-lag]) ** 2))
            # Only include non-zero, positive values
            if std > 1e-10:  # Avoid zero or very small values
                tau.append(std)
                valid_lags.append(lag)

        if len(tau) < 2:
            return 0.5

        tau = np.array(tau)
        lags_array = np.array(valid_lags)

        # Ensure all values are positive before taking log
        if np.any(lags_array <= 0) or np.any(tau <= 0):
            return 0.5

        try:
            # Fit log(tau) vs log(lag)
            poly = np.polyfit(np.log(lags_array), np.log(tau), 1)
            hurst = poly[0]
            return np.clip(hurst, 0, 1)
        except (ValueError, np.linalg.LinAlgError):
            # If fitting fails, return neutral value
            return 0.5

    def select_strategy(self, market_data: Dict) -> Tuple[str, float]:
        """
        Select the best strategy based on current conditions

        Args:
            market_data: Current market data

        Returns:
            (strategy_name, confidence_score)
        """
        # Update price history
        if 'price' in market_data:
            self.price_history.append(market_data['price'])
        if 'volume' in market_data:
            self.volume_history.append(market_data['volume'])

        # Detect regime
        self.current_regime = self.detect_market_regime()

        # Get candidate strategies for this regime
        regime_type = self.current_regime.regime_type
        if regime_type in self.regime_strategy_map:
            candidates = self.regime_strategy_map[regime_type]
        else:
            candidates = list(self.strategies.keys())

        # Score each candidate
        scores = {}
        for strategy_name in candidates:
            if strategy_name not in self.strategies:
                continue

            perf = self.performance[strategy_name]

            # If not enough trades, use default confidence
            if perf.total_trades < self.min_trades_for_selection:
                # For untested strategies, give them a base score that allows initial trading
                # This ensures we can start trading even without historical performance
                base_score = 0.4  # Higher base score for untested strategies
                
                # If regime is unknown, still allow trading (don't penalize too much)
                if self.current_regime.regime_type == 'unknown':
                    score = base_score  # Don't reduce for unknown regime
                else:
                    # Boost score based on regime confidence, but ensure minimum
                    score = base_score * (0.6 + 0.4 * self.current_regime.confidence)
            else:
                score = perf.get_score()
            # Boost score based on regime confidence
            score *= (0.5 + 0.5 * self.current_regime.confidence)

            scores[strategy_name] = score

        if not scores:
            # Fallback to first available strategy
            return list(self.strategies.keys())[0], 0.5

        # Select best strategy
        best_strategy = max(scores.items(), key=lambda x: x[1])
        self.active_strategy = best_strategy[0]

        logger.info(f"Selected strategy: {best_strategy[0]} "
                   f"(score: {best_strategy[1]:.3f}, "
                   f"regime: {self.current_regime.regime_type})")

        return best_strategy

    def update_performance(self, strategy_name: str, pnl: float, timestamp: datetime):
        """Update strategy performance after trade execution"""
        if strategy_name in self.performance:
            self.performance[strategy_name].update(pnl, timestamp)

    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of all strategy performances"""
        data = []
        for name, perf in self.performance.items():
            data.append({
                'Strategy': name,
                'Total Trades': perf.total_trades,
                'Win Rate': f"{perf.get_win_rate():.2%}",
                'Total PnL': f"${perf.total_pnl:.2f}",
                'Sharpe Ratio': f"{perf.get_sharpe_ratio():.2f}",
                'Score': f"{perf.get_score():.3f}"
            })
        return pd.DataFrame(data)

    def get_regime_info(self) -> Dict:
        """Get current market regime information"""
        if self.current_regime:
            return {
                'regime': self.current_regime.regime_type,
                'confidence': f"{self.current_regime.confidence:.2%}",
                'volatility': f"{self.current_regime.volatility:.2%}",
                'trend_strength': f"{self.current_regime.trend_strength:.3f}",
                'timestamp': self.current_regime.timestamp
            }
        return {'regime': 'unknown'}

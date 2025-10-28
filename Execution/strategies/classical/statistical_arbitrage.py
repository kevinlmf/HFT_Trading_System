"""
Statistical Arbitrage Strategy for HFT
Multi-asset mean reversion based on cointegration and PCA
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA


@dataclass
class StatArbParams:
    """Statistical arbitrage parameters"""
    lookback_period: int = 60  # Period for calculating statistics
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5  # Z-score threshold for exit
    stop_loss: float = 3.0  # Z-score stop loss
    max_holding_period: int = 100  # Maximum holding period
    min_half_life: int = 5  # Minimum half-life for mean reversion
    max_half_life: int = 50  # Maximum half-life for mean reversion
    position_size: float = 1.0  # Position size scaling


class StatisticalArbitrageStrategy:
    """
    Statistical arbitrage using cointegration and PCA

    Features:
    - Multi-asset portfolio construction
    - Cointegration testing
    - PCA-based factor extraction
    - Mean reversion signals with z-score
    - Half-life estimation for timing
    """

    def __init__(self, params: Optional[StatArbParams] = None):
        self.params = params or StatArbParams()
        self.positions: Dict[str, float] = {}
        self.entry_zscores: Dict[str, float] = {}
        self.holding_periods: Dict[str, int] = {}
        self.portfolio_weights: Optional[np.ndarray] = None
        self.pca_model: Optional[PCA] = None

    def reset(self):
        """Reset strategy state"""
        self.positions = {}
        self.entry_zscores = {}
        self.holding_periods = {}
        self.portfolio_weights = None
        self.pca_model = None

    def find_cointegrated_pairs(
        self,
        price_data: Dict[str, pd.DataFrame],
        p_value_threshold: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated asset pairs using Engle-Granger test

        Returns:
            List of (asset1, asset2, p_value) for cointegrated pairs
        """
        from statsmodels.tsa.stattools import coint

        symbols = list(price_data.keys())
        cointegrated_pairs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]

                # Get price series
                prices1 = self._get_close_prices(price_data[sym1])
                prices2 = self._get_close_prices(price_data[sym2])

                if prices1 is None or prices2 is None:
                    continue

                # Align series
                aligned = pd.DataFrame({'p1': prices1, 'p2': prices2}).dropna()
                if len(aligned) < self.params.lookback_period:
                    continue

                try:
                    # Perform cointegration test
                    score, p_value, _ = coint(aligned['p1'], aligned['p2'])

                    if p_value < p_value_threshold:
                        cointegrated_pairs.append((sym1, sym2, p_value))
                except Exception:
                    continue

        return sorted(cointegrated_pairs, key=lambda x: x[2])  # Sort by p-value

    def calculate_spread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> Tuple[pd.Series, float]:
        """
        Calculate spread between two assets

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            hedge_ratio: Pre-calculated hedge ratio (optional)

        Returns:
            (spread, hedge_ratio)
        """
        # Align series
        aligned = pd.DataFrame({'p1': prices1, 'p2': prices2}).dropna()

        if hedge_ratio is None:
            # Calculate hedge ratio using linear regression
            from sklearn.linear_model import LinearRegression
            X = aligned['p2'].values.reshape(-1, 1)
            y = aligned['p1'].values
            model = LinearRegression()
            model.fit(X, y)
            hedge_ratio = model.coef_[0]

        # Calculate spread
        spread = aligned['p1'] - hedge_ratio * aligned['p2']

        return spread, hedge_ratio

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process

        Returns:
            Half-life in periods (e.g., days)
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag

        # Drop NaN
        data = pd.DataFrame({'spread': spread, 'spread_lag': spread_lag, 'diff': spread_diff}).dropna()

        if len(data) < 10:
            return np.inf

        try:
            # Fit AR(1) model: delta_spread = lambda * (spread - mean)
            from sklearn.linear_model import LinearRegression
            X = data['spread_lag'].values.reshape(-1, 1)
            y = data['diff'].values
            model = LinearRegression()
            model.fit(X, y)

            lambda_param = -model.coef_[0]

            if lambda_param <= 0:
                return np.inf

            half_life = -np.log(2) / lambda_param
            return half_life

        except Exception:
            return np.inf

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate z-score of spread

        Args:
            spread: Spread series
            lookback: Lookback period (uses params.lookback_period if None)

        Returns:
            Z-score series
        """
        lookback = lookback or self.params.lookback_period

        mean = spread.rolling(lookback).mean()
        std = spread.rolling(lookback).std()

        zscore = (spread - mean) / std
        return zscore

    def extract_factors_pca(
        self,
        price_data: Dict[str, pd.DataFrame],
        n_components: int = 3
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract common factors using PCA

        Returns:
            (factor_loadings DataFrame, explained_variance_ratio)
        """
        # Build price matrix
        price_matrix = {}
        for symbol, df in price_data.items():
            prices = self._get_close_prices(df)
            if prices is not None:
                price_matrix[symbol] = prices

        if len(price_matrix) < 2:
            raise ValueError("Need at least 2 assets for PCA")

        # Create aligned DataFrame
        price_df = pd.DataFrame(price_matrix).dropna()

        # Calculate returns
        returns = price_df.pct_change().dropna()

        # Fit PCA
        self.pca_model = PCA(n_components=min(n_components, len(returns.columns)))
        self.pca_model.fit(returns)

        # Get factor loadings
        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            columns=[f'Factor_{i+1}' for i in range(self.pca_model.n_components_)],
            index=returns.columns
        )

        return loadings, self.pca_model.explained_variance_ratio_

    def generate_signals_pair(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        symbol_pair: Tuple[str, str]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals for a pair

        Returns:
            (signals_asset1, signals_asset2) where signals are -1, 0, or 1
        """
        # Calculate spread and z-score
        spread, hedge_ratio = self.calculate_spread(prices1, prices2)
        zscore = self.calculate_zscore(spread)

        # Check half-life
        half_life = self.calculate_half_life(spread)
        if not (self.params.min_half_life <= half_life <= self.params.max_half_life):
            # Mean reversion too fast or too slow
            return pd.Series(0, index=prices1.index), pd.Series(0, index=prices2.index)

        # Generate signals
        signals1 = pd.Series(0, index=prices1.index)
        signals2 = pd.Series(0, index=prices2.index)

        for i in range(len(zscore)):
            if pd.isna(zscore.iloc[i]):
                continue

            z = zscore.iloc[i]
            idx = zscore.index[i]

            # Check if we have an open position
            pair_key = f"{symbol_pair[0]}_{symbol_pair[1]}"
            is_long = pair_key in self.positions and self.positions[pair_key] > 0
            is_short = pair_key in self.positions and self.positions[pair_key] < 0

            # Entry signals
            if not is_long and not is_short:
                if z > self.params.entry_threshold:
                    # Spread too high -> short spread (short asset1, long asset2)
                    signals1.loc[idx] = -1
                    signals2.loc[idx] = 1
                    self.positions[pair_key] = -1
                    self.entry_zscores[pair_key] = z
                    self.holding_periods[pair_key] = 0

                elif z < -self.params.entry_threshold:
                    # Spread too low -> long spread (long asset1, short asset2)
                    signals1.loc[idx] = 1
                    signals2.loc[idx] = -1
                    self.positions[pair_key] = 1
                    self.entry_zscores[pair_key] = z
                    self.holding_periods[pair_key] = 0

            # Exit signals
            else:
                self.holding_periods[pair_key] += 1

                # Exit conditions
                should_exit = False

                # Mean reversion exit
                if abs(z) < self.params.exit_threshold:
                    should_exit = True

                # Stop loss
                if abs(z) > self.params.stop_loss:
                    should_exit = True

                # Maximum holding period
                if self.holding_periods[pair_key] >= self.params.max_holding_period:
                    should_exit = True

                if should_exit:
                    # Close position (opposite signals)
                    if is_long:
                        signals1.loc[idx] = -1  # Sell asset1
                        signals2.loc[idx] = 1  # Buy back asset2
                    else:
                        signals1.loc[idx] = 1  # Buy back asset1
                        signals2.loc[idx] = -1  # Sell asset2

                    # Clear position
                    del self.positions[pair_key]
                    del self.entry_zscores[pair_key]
                    del self.holding_periods[pair_key]

        return signals1, signals2

    def _get_close_prices(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract close prices from DataFrame"""
        if 'close' in df.columns:
            return df['close']
        elif 'Close' in df.columns:
            return df['Close']
        else:
            return None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for single asset (for compatibility)

        For multi-asset strategy, use generate_signals_pair instead
        """
        # Simple mean reversion on single asset
        if 'close' not in data.columns:
            return pd.Series(0, index=data.index)

        prices = data['close']
        mean = prices.rolling(self.params.lookback_period).mean()
        std = prices.rolling(self.params.lookback_period).std()
        zscore = (prices - mean) / std

        signals = pd.Series(0, index=data.index)
        signals[zscore > self.params.entry_threshold] = -1
        signals[zscore < -self.params.entry_threshold] = 1

        return signals


# Example usage
if __name__ == "__main__":
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')

    # Generate cointegrated series
    common_trend = np.cumsum(np.random.randn(200)) * 0.1 + 100
    prices1 = common_trend + np.random.randn(200) * 2
    prices2 = common_trend * 1.5 + np.random.randn(200) * 3

    price_data = {
        'ASSET1': pd.DataFrame({'close': prices1}, index=dates),
        'ASSET2': pd.DataFrame({'close': prices2}, index=dates)
    }

    # Create strategy
    strategy = StatisticalArbitrageStrategy()

    # Find cointegrated pairs
    pairs = strategy.find_cointegrated_pairs(price_data)
    print(f"Found {len(pairs)} cointegrated pairs:")
    for sym1, sym2, pval in pairs:
        print(f"  {sym1} - {sym2}: p-value = {pval:.4f}")

    # Generate signals
    if pairs:
        sym1, sym2, _ = pairs[0]
        signals1, signals2 = strategy.generate_signals_pair(
            price_data[sym1]['close'],
            price_data[sym2]['close'],
            (sym1, sym2)
        )
        print(f"\nGenerated {signals1.abs().sum()} trades for {sym1}")
        print(f"Generated {signals2.abs().sum()} trades for {sym2}")

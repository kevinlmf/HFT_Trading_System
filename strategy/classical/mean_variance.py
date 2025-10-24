import numpy as np
import pandas as pd

class MeanVarianceStrategy:
    def __init__(self, risk_aversion=1.0):
        self.risk_aversion = risk_aversion

    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        mu = returns.mean().values.reshape(-1, 1)
        Sigma = returns.cov().values
        inv_Sigma = np.linalg.pinv(Sigma)
        weights = inv_Sigma @ mu / (self.risk_aversion + 1e-8)
        weights /= weights.sum()
        return weights.flatten()

    def decide_position(self, prices: pd.DataFrame, portfolio_value: float):
        returns = prices.pct_change().dropna()
        weights = self.compute_weights(returns)
        latest_prices = prices.iloc[-1].values
        shares = portfolio_value * weights / latest_prices
        return dict(zip(prices.columns, shares))

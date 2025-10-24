"""
Advanced Portfolio Management System
Handles position sizing, risk management, and portfolio optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.optimize import minimize
from scipy import stats
warnings.filterwarnings('ignore')

class RiskModel(Enum):
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hrp"

@dataclass
class RiskConstraints:
    """Risk constraints for portfolio construction"""
    max_position_size: float = 0.2  # Maximum single position weight
    max_sector_exposure: float = 0.4  # Maximum sector exposure
    max_correlation_exposure: float = 0.6  # Max exposure to highly correlated assets
    max_portfolio_volatility: float = 0.15  # Maximum portfolio volatility
    max_drawdown_limit: float = 0.05  # Maximum allowed drawdown
    var_limit: float = 0.02  # Value at Risk limit (daily)
    concentration_limit: float = 0.5  # Max weight in top 5 positions

@dataclass
class PortfolioState:
    """Current portfolio state"""
    positions: Dict[str, float]  # symbol -> quantity
    weights: Dict[str, float]  # symbol -> weight
    cash: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    
class AdvancedPortfolioManager:
    """
    Advanced portfolio management with multiple risk models and constraints
    
    Features:
    - Multiple portfolio construction methods
    - Dynamic risk budgeting
    - Sector and correlation constraints
    - Real-time risk monitoring
    - Position sizing with Kelly criterion
    - Transaction cost optimization
    """
    
    def __init__(self, 
                 initial_capital: float,
                 risk_model: RiskModel = RiskModel.RISK_PARITY,
                 constraints: RiskConstraints = None,
                 lookback_period: int = 252,
                 rebalance_threshold: float = 0.05,
                 transaction_cost_bps: float = 10):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital
            risk_model: Risk model for portfolio construction
            constraints: Risk constraints
            lookback_period: Days of historical data for calculations
            rebalance_threshold: Threshold for rebalancing (weight deviation)
            transaction_cost_bps: Transaction costs in basis points
        """
        self.initial_capital = initial_capital
        self.risk_model = risk_model
        self.constraints = constraints or RiskConstraints()
        self.lookback_period = lookback_period
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost_bps = transaction_cost_bps
        
        # Portfolio state
        self.state = PortfolioState(
            positions={},
            weights={},
            cash=initial_capital,
            total_value=initial_capital,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        
        # Risk monitoring
        self.risk_metrics_history = []
        self.rebalance_history = []
        
        # Asset metadata
        self.asset_sectors = {}  # symbol -> sector mapping
        self.asset_correlations = {}  # Correlation matrix cache
    
    def calculate_optimal_weights(self, 
                                price_data: Dict[str, pd.DataFrame],
                                expected_returns: Optional[Dict[str, float]] = None,
                                risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights based on selected risk model
        
        Args:
            price_data: Historical price data for assets
            expected_returns: Expected returns (if None, will estimate)
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dictionary of symbol -> optimal weight
        """
        # Prepare data
        returns_data = self._prepare_returns_data(price_data)
        if returns_data.empty:
            return {}
        
        # Estimate expected returns if not provided
        if expected_returns is None:
            expected_returns = self._estimate_expected_returns(returns_data)
        
        # Calculate covariance matrix
        cov_matrix = returns_data.cov().values
        
        # Apply selected risk model
        if self.risk_model == RiskModel.EQUAL_WEIGHT:
            weights = self._equal_weight(list(returns_data.columns))
        elif self.risk_model == RiskModel.INVERSE_VOLATILITY:
            weights = self._inverse_volatility(returns_data)
        elif self.risk_model == RiskModel.MEAN_VARIANCE:
            weights = self._mean_variance_optimization(expected_returns, cov_matrix, risk_aversion)
        elif self.risk_model == RiskModel.RISK_PARITY:
            weights = self._risk_parity(cov_matrix, list(returns_data.columns))
        elif self.risk_model == RiskModel.BLACK_LITTERMAN:
            weights = self._black_litterman(returns_data, expected_returns, cov_matrix)
        elif self.risk_model == RiskModel.HIERARCHICAL_RISK_PARITY:
            weights = self._hierarchical_risk_parity(returns_data)
        else:
            weights = self._equal_weight(list(returns_data.columns))
        
        # Apply constraints
        weights = self._apply_constraints(weights, returns_data)
        
        return weights
    
    def _prepare_returns_data(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare aligned returns data"""
        returns_dict = {}
        
        for symbol, df in price_data.items():
            if 'close' in df.columns and len(df) > 20:
                returns = df['close'].pct_change().dropna()
                if len(returns) > 0:
                    returns_dict[symbol] = returns
        
        if not returns_dict:
            return pd.DataFrame()
        
        # Align all returns data
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        # Use only recent data if too much history
        if len(returns_df) > self.lookback_period:
            returns_df = returns_df.tail(self.lookback_period)
        
        return returns_df
    
    def _estimate_expected_returns(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Estimate expected returns using various methods"""
        expected_returns = {}
        
        for symbol in returns_data.columns:
            returns = returns_data[symbol].dropna()
            
            if len(returns) < 20:
                expected_returns[symbol] = 0.08 / 252  # Default daily return
                continue
            
            # Use combination of historical mean and momentum
            historical_mean = returns.mean()
            
            # Recent momentum (last 60 days vs previous 60 days)
            if len(returns) >= 120:
                recent_returns = returns.tail(60).mean()
                previous_returns = returns.iloc[-120:-60].mean()
                momentum_factor = recent_returns - previous_returns
            else:
                momentum_factor = 0
            
            # Combine with 70% weight on historical, 30% on momentum
            expected_returns[symbol] = 0.7 * historical_mean + 0.3 * momentum_factor
        
        return expected_returns
    
    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Equal weight portfolio"""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    def _inverse_volatility(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Inverse volatility weighting"""
        volatilities = returns_data.std()
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()
    
    def _mean_variance_optimization(self, expected_returns: Dict[str, float], 
                                   cov_matrix: np.ndarray, 
                                   risk_aversion: float) -> Dict[str, float]:
        """Mean-variance optimization"""
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Convert expected returns to array
        mu = np.array([expected_returns[symbol] for symbol in symbols])
        
        # Objective function: maximize utility = expected return - 0.5 * risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (0 to max position size)
        bounds = [(0, self.constraints.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                return {symbols[i]: result.x[i] for i in range(n_assets)}
        except Exception as e:
            print(f"Mean-variance optimization failed: {e}")
        
        # Fallback to equal weight
        return self._equal_weight(symbols)
    
    def _risk_parity(self, cov_matrix: np.ndarray, symbols: List[str]) -> Dict[str, float]:
        """Risk parity portfolio (equal risk contribution)"""
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        def risk_budget_objective(weights):
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            return np.sum((contrib - 1.0/n_assets) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.constraints.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(risk_budget_objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            if result.success:
                return {symbols[i]: result.x[i] for i in range(n_assets)}
        except Exception as e:
            print(f"Risk parity optimization failed: {e}")
        
        return self._equal_weight(symbols)
    
    def _black_litterman(self, returns_data: pd.DataFrame, 
                        expected_returns: Dict[str, float], 
                        cov_matrix: np.ndarray) -> Dict[str, float]:
        """Black-Litterman model (simplified implementation)"""
        # This is a simplified version - full implementation would require views and confidence levels
        symbols = list(returns_data.columns)
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Use market cap weights as prior (simplified to equal weight)
        prior_weights = np.ones(n_assets) / n_assets
        
        # Implied returns from CAPM
        market_var = 0.15 ** 2  # Assume 15% market volatility
        implied_returns = market_var * np.dot(cov_matrix, prior_weights)
        
        # Combine with views (simplified - just use expected returns)
        mu = np.array([expected_returns[symbol] for symbol in symbols])
        
        # Black-Litterman returns (simplified)
        tau = 0.05  # Scaling factor
        bl_returns = implied_returns + tau * (mu - implied_returns)
        
        # Optimize with Black-Litterman returns
        return self._mean_variance_optimization(
            {symbols[i]: bl_returns[i] for i in range(n_assets)}, 
            cov_matrix, 
            1.0
        )
    
    def _hierarchical_risk_parity(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Hierarchical Risk Parity (simplified version)"""
        # This is a basic implementation - full HRP requires distance matrix and clustering
        symbols = list(returns_data.columns)
        
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        # Inverse volatility as base weights
        vol_weights = self._inverse_volatility(returns_data)
        
        # Adjust for correlation (simplified approach)
        adjusted_weights = {}
        for symbol in symbols:
            # Reduce weight if highly correlated with others
            avg_correlation = corr_matrix[symbol].drop(symbol).abs().mean()
            correlation_penalty = 1.0 - min(avg_correlation * 0.5, 0.5)
            adjusted_weights[symbol] = vol_weights[symbol] * correlation_penalty
        
        # Normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _apply_constraints(self, weights: Dict[str, float], 
                          returns_data: pd.DataFrame) -> Dict[str, float]:
        """Apply portfolio constraints"""
        if not weights:
            return weights
        
        # Max position size constraint
        total_weight = sum(weights.values())
        for symbol in weights:
            if weights[symbol] > self.constraints.max_position_size:
                weights[symbol] = self.constraints.max_position_size
        
        # Renormalize after position size constraints
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Concentration limit (top 5 positions)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_5_weight = sum(w for _, w in sorted_weights[:5])
        if top_5_weight > self.constraints.concentration_limit:
            # Scale down top positions
            scale_factor = self.constraints.concentration_limit / top_5_weight
            for i in range(min(5, len(sorted_weights))):
                symbol = sorted_weights[i][0]
                weights[symbol] *= scale_factor
            
            # Renormalize
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Portfolio volatility constraint
        if len(returns_data) > 20:
            cov_matrix = returns_data.cov()
            weight_array = np.array([weights.get(col, 0) for col in returns_data.columns])
            portfolio_vol = np.sqrt(np.dot(weight_array, np.dot(cov_matrix.values, weight_array))) * np.sqrt(252)
            
            if portfolio_vol > self.constraints.max_portfolio_volatility:
                # Scale down all weights to meet volatility constraint
                scale_factor = self.constraints.max_portfolio_volatility / portfolio_vol
                weights = {k: v * scale_factor for k, v in weights.items()}
        
        return weights
    
    def should_rebalance(self, current_weights: Dict[str, float], 
                        target_weights: Dict[str, float]) -> bool:
        """Determine if portfolio should be rebalanced"""
        if not current_weights or not target_weights:
            return True
        
        # Check weight deviations
        total_deviation = 0
        for symbol in target_weights:
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights[symbol]
            deviation = abs(current_w - target_w)
            total_deviation += deviation
        
        return total_deviation > self.rebalance_threshold
    
    def calculate_rebalance_trades(self, current_positions: Dict[str, float],
                                 target_weights: Dict[str, float],
                                 current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate trades needed for rebalancing
        
        Returns:
            Dictionary of symbol -> trade quantity (+ for buy, - for sell)
        """
        if not target_weights or not current_prices:
            return {}
        
        trades = {}
        total_portfolio_value = self.state.total_value
        
        for symbol in set(list(current_positions.keys()) + list(target_weights.keys())):
            current_quantity = current_positions.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            current_price = current_prices.get(symbol, 0)
            
            if current_price <= 0:
                continue
            
            # Calculate target quantity
            target_value = total_portfolio_value * target_weight
            target_quantity = target_value / current_price
            
            # Calculate trade needed
            trade_quantity = target_quantity - current_quantity
            
            if abs(trade_quantity) > 0.01:  # Minimum trade size
                trades[symbol] = trade_quantity
        
        # Optimize trades for transaction costs
        trades = self._optimize_trades_for_costs(trades, current_prices)
        
        return trades
    
    def _optimize_trades_for_costs(self, trades: Dict[str, float], 
                                  current_prices: Dict[str, float]) -> Dict[str, float]:
        """Optimize trades to minimize transaction costs"""
        # Simple optimization: eliminate small trades that are expensive relative to benefit
        optimized_trades = {}
        
        for symbol, quantity in trades.items():
            if symbol not in current_prices:
                continue
                
            trade_value = abs(quantity) * current_prices[symbol]
            transaction_cost = trade_value * self.transaction_cost_bps / 10000.0
            
            # Only execute if trade value exceeds minimum threshold
            if trade_value > transaction_cost * 10:  # 10x transaction cost minimum
                optimized_trades[symbol] = quantity
        
        return optimized_trades
    
    def update_portfolio_state(self, current_positions: Dict[str, float],
                              current_prices: Dict[str, float],
                              cash: float,
                              realized_pnl: float = 0.0):
        """Update current portfolio state"""
        # Calculate total value
        position_value = sum(positions * current_prices.get(symbol, 0) 
                           for symbol, positions in current_positions.items())
        total_value = position_value + cash
        
        # Calculate weights
        weights = {}
        if total_value > 0:
            for symbol, quantity in current_positions.items():
                if symbol in current_prices and quantity != 0:
                    weights[symbol] = (quantity * current_prices[symbol]) / total_value
        
        # Update state
        self.state.positions = current_positions.copy()
        self.state.weights = weights
        self.state.cash = cash
        self.state.total_value = total_value
        self.state.unrealized_pnl = total_value - self.initial_capital - realized_pnl
        self.state.realized_pnl = realized_pnl
    
    def calculate_portfolio_risk_metrics(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics"""
        if not self.state.weights or returns_data.empty:
            return {}
        
        # Portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        for symbol, weight in self.state.weights.items():
            if symbol in returns_data.columns:
                portfolio_returns += weight * returns_data[symbol]
        
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) < 20:
            return {}
        
        # Risk metrics
        metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
        }
        
        # Store for history
        self.risk_metrics_history.append({
            'timestamp': returns_data.index[-1] if len(returns_data) > 0 else pd.Timestamp.now(),
            **metrics
        })
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        return {
            'total_value': self.state.total_value,
            'cash': self.state.cash,
            'positions': dict(self.state.positions),
            'weights': dict(self.state.weights),
            'unrealized_pnl': self.state.unrealized_pnl,
            'realized_pnl': self.state.realized_pnl,
            'total_return': (self.state.total_value - self.initial_capital) / self.initial_capital,
            'number_of_positions': len([w for w in self.state.weights.values() if abs(w) > 0.001]),
            'largest_position': max(self.state.weights.values()) if self.state.weights else 0,
            'concentration_top_5': sum(sorted(self.state.weights.values(), reverse=True)[:5]),
            'risk_model': self.risk_model.value,
            'last_rebalance': self.rebalance_history[-1] if self.rebalance_history else None
        }
    
    def check_risk_violations(self, risk_metrics: Dict[str, float]) -> List[str]:
        """Check for risk constraint violations"""
        violations = []
        
        if risk_metrics.get('volatility', 0) > self.constraints.max_portfolio_volatility:
            violations.append(f"Portfolio volatility ({risk_metrics['volatility']:.2%}) exceeds limit ({self.constraints.max_portfolio_volatility:.2%})")
        
        if abs(risk_metrics.get('var_95', 0)) > self.constraints.var_limit:
            violations.append(f"VaR ({abs(risk_metrics['var_95']):.2%}) exceeds limit ({self.constraints.var_limit:.2%})")
        
        # Check position size constraints
        if self.state.weights:
            max_weight = max(self.state.weights.values())
            if max_weight > self.constraints.max_position_size:
                violations.append(f"Position size ({max_weight:.2%}) exceeds limit ({self.constraints.max_position_size:.2%})")
            
            # Check concentration
            top_5_weight = sum(sorted(self.state.weights.values(), reverse=True)[:5])
            if top_5_weight > self.constraints.concentration_limit:
                violations.append(f"Top 5 concentration ({top_5_weight:.2%}) exceeds limit ({self.constraints.concentration_limit:.2%})")
        
        return violations
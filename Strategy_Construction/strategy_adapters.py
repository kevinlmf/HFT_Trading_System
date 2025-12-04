"""
策略适配器 - 将现有策略适配到StrategyRegistry

解决策略注册问题，让现有策略可以通过registry访问
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Strategy_Construction.base_strategy import BaseStrategy, StrategyMetadata, StrategyCategory
from Strategy_Construction.strategy_registry import StrategyRegistry
from Strategy_Construction.classical.momentum_strategy import MomentumStrategy
from Strategy_Construction.classical.statistical_arbitrage import StatisticalArbitrageStrategy
from Strategy_Construction.classical.pairs_trading import PairsTradingStrategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


# ========== Momentum Strategy Adapter ==========

@StrategyRegistry.register
class MomentumStrategyAdapter(BaseStrategy):
    """Momentum策略适配器"""
    
    metadata = StrategyMetadata(
        name="MomentumStrategy",
        category=StrategyCategory.TREND,
        description="Multi-factor momentum strategy combining price, technical, and cross-sectional momentum",
        target_frequency="minute",
        capital_requirement=100000.0,
        risk_level="medium",
        sharpe_expectation=1.5,
        max_drawdown_tolerance=0.15,
        theory_base="Momentum effect (Jegadeesh & Titman, 1993)",
        key_papers=["Jegadeesh & Titman (1993) - Returns to Buying Winners and Selling Losers"],
        applicable_markets=["stock", "futures", "crypto"]
    )
    
    def __init__(self, metadata: StrategyMetadata = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(metadata or self.metadata, config)
        # 创建底层策略实例
        # 使用更短的lookback周期以适应实时交易（不需要252天的数据）
        # 降低到[3, 5, 10]以适应Yahoo Finance等较慢的数据源（只需要10个数据点）
        default_lookback = config.get('lookback_periods', [3, 5, 10]) if config else [3, 5, 10]
        self.momentum_strategy = MomentumStrategy(
            lookback_periods=default_lookback,
            volatility_adjustment=config.get('volatility_adjustment', True) if config else True,
            cross_sectional_ranking=config.get('cross_sectional_ranking', False) if config else False,  # 单标的时禁用
            max_position_size=config.get('max_position_size', 0.1) if config else 0.1,
            min_momentum_threshold=config.get('min_momentum_threshold', 0.01) if config else 0.01,  # 降低阈值
        )
        self.current_price_data = {}
    
    def initialize(self, historical_data: pd.DataFrame) -> None:
        """初始化策略"""
        self.is_initialized = True
        self.current_data = historical_data
    
    def on_data(self, data: pd.DataFrame) -> None:
        """数据更新回调"""
        self.current_data = data
        # 更新价格数据字典
        if 'symbol' in data.columns:
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                self.current_price_data[symbol] = symbol_data
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        if data is None or data.empty:
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasoning': 'No data'}
        
        # 准备价格数据字典
        price_data = {}
        if 'symbol' in data.columns:
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].copy()
                if 'close' not in symbol_data.columns and 'last_price' in symbol_data.columns:
                    symbol_data['close'] = symbol_data['last_price']
                price_data[symbol] = symbol_data
        else:
            # 单symbol情况
            symbol_data = data.copy()
            if 'close' not in symbol_data.columns and 'last_price' in symbol_data.columns:
                symbol_data['close'] = symbol_data['last_price']
            price_data['DEFAULT'] = symbol_data
        
        # 生成信号
        signals = self.momentum_strategy.generate_signals(price_data)
        
        # 转换为标准格式
        if not signals:
            # 检查为什么没有信号
            for sym, sym_df in price_data.items():
                required_len = max(self.momentum_strategy.lookback_periods)
                actual_len = len(sym_df)
                has_close = 'close' in sym_df.columns if not sym_df.empty else False
                logger.info(f"⚠️  No signals for {sym}: have {actual_len} data points, need {required_len}, "
                           f"has_close={has_close}")
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 
                   'reasoning': f'No signals (need {max(self.momentum_strategy.lookback_periods)} data points, have {len(list(price_data.values())[0]) if price_data else 0})'}
        
        # 取第一个symbol的信号
        first_symbol = list(signals.keys())[0]
        symbol_signals = signals[first_symbol]
        
        if not symbol_signals:
            logger.info(f"⚠️  Empty signal list for {first_symbol} (strategy returned empty list)")
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasoning': 'Empty signal list'}
        
        # 计算平均强度和置信度
        avg_strength = np.mean([s.strength for s in symbol_signals])
        avg_confidence = np.mean([s.confidence for s in symbol_signals])
        
        # 确定动作
        if avg_strength > 0.1:
            action = 'BUY'
        elif avg_strength < -0.1:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'strength': float(abs(avg_strength)),
            'confidence': float(avg_confidence),
            'reasoning': f'Momentum signals: {len(symbol_signals)} signals, avg strength: {avg_strength:.3f}',
            'indicators': {
                'momentum_strength': float(avg_strength),
                'confidence': float(avg_confidence),
                'num_signals': len(symbol_signals)
            }
        }
    
    def calculate_position(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """计算目标头寸"""
        if signal['action'] == 'HOLD':
            return {
                'target_position': self.position,
                'position_change': 0.0,
                'order_type': 'MARKET',
                'limit_price': None,
                'stop_loss': None,
                'take_profit': None
            }
        
        # 根据信号强度计算目标头寸
        strength = signal['strength']
        confidence = signal['confidence']
        
        # 目标仓位 = 强度 * 置信度 * 最大仓位
        max_position_size = self.config.get('max_position_size', 0.1)
        target_position_pct = strength * confidence * max_position_size
        
        if signal['action'] == 'SELL':
            target_position_pct = -target_position_pct
        
        # 转换为绝对数量（假设使用portfolio value）
        portfolio_value = self.portfolio_value
        target_value = portfolio_value * target_position_pct
        target_position = target_value / current_price if current_price > 0 else 0
        
        position_change = target_position - self.position
        
        return {
            'target_position': float(target_position),
            'position_change': float(position_change),
            'order_type': 'MARKET',
            'limit_price': None,
            'stop_loss': current_price * 0.95 if signal['action'] == 'BUY' else None,
            'take_profit': current_price * 1.1 if signal['action'] == 'BUY' else None
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """获取策略性能指标"""
        if len(self.trades) < 2:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # 计算收益
        returns = []
        for i in range(1, len(self.trades)):
            if self.trades[i-1]['type'] == 'BUY' and self.trades[i]['type'] == 'SELL':
                pnl = (self.trades[i]['price'] - self.trades[i-1]['price']) / self.trades[i-1]['price']
                returns.append(pnl)
        
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        returns = np.array(returns)
        equity_curve = np.cumprod(1 + returns)
        
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'total_return': float(np.prod(1 + returns) - 1),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': float(np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))) if np.sum(returns[returns < 0]) != 0 else 0.0
        }


# ========== Mean Reversion Strategy Adapter ==========

@StrategyRegistry.register
class MeanReversionStrategyAdapter(BaseStrategy):
    """Mean Reversion策略适配器（简化版）"""
    
    metadata = StrategyMetadata(
        name="MeanReversionStrategy",
        category=StrategyCategory.MEAN_REVERSION,
        description="Mean reversion strategy based on price deviation from moving average",
        target_frequency="minute",
        capital_requirement=100000.0,
        risk_level="medium",
        sharpe_expectation=1.2,
        max_drawdown_tolerance=0.12,
        theory_base="Mean reversion theory",
        key_papers=["Mean Reversion in Stock Prices (Poterba & Summers, 1988)"],
        applicable_markets=["stock", "futures", "crypto"]
    )
    
    def __init__(self, metadata: StrategyMetadata = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(metadata or self.metadata, config)
        self.lookback_period = config.get('lookback_period', 20) if config else 20
        self.deviation_threshold = config.get('deviation_threshold', 0.02) if config else 0.02
    
    def initialize(self, historical_data: pd.DataFrame) -> None:
        """初始化策略"""
        self.is_initialized = True
        self.current_data = historical_data
    
    def on_data(self, data: pd.DataFrame) -> None:
        """数据更新回调"""
        self.current_data = data
    
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        if data is None or data.empty or len(data) < self.lookback_period:
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasoning': 'Insufficient data'}
        
        # 获取价格
        if 'close' in data.columns:
            prices = data['close']
        elif 'last_price' in data.columns:
            prices = data['last_price']
        else:
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasoning': 'No price data'}
        
        prices = pd.Series(prices).astype(float)
        current_price = prices.iloc[-1]
        
        # 计算移动平均
        ma = prices.rolling(self.lookback_period).mean().iloc[-1]
        
        if pd.isna(ma):
            return {'action': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasoning': 'MA not available'}
        
        # 计算偏离度
        deviation = (current_price - ma) / ma
        
        # 生成信号
        if deviation < -self.deviation_threshold:
            # 价格低于均值，买入
            action = 'BUY'
            strength = min(abs(deviation) / self.deviation_threshold, 1.0)
        elif deviation > self.deviation_threshold:
            # 价格高于均值，卖出
            action = 'SELL'
            strength = min(deviation / self.deviation_threshold, 1.0)
        else:
            action = 'HOLD'
            strength = 0.0
        
        confidence = min(abs(deviation) / self.deviation_threshold, 1.0)
        
        return {
            'action': action,
            'strength': float(strength),
            'confidence': float(confidence),
            'reasoning': f'Price deviation: {deviation:.3f}, MA: {ma:.2f}, Price: {current_price:.2f}',
            'indicators': {
                'deviation': float(deviation),
                'ma': float(ma),
                'current_price': float(current_price)
            }
        }
    
    def calculate_position(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """计算目标头寸"""
        if signal['action'] == 'HOLD':
            return {
                'target_position': self.position,
                'position_change': 0.0,
                'order_type': 'MARKET',
                'limit_price': None,
                'stop_loss': None,
                'take_profit': None
            }
        
        max_position_size = self.config.get('max_position_size', 0.1)
        target_position_pct = signal['strength'] * signal['confidence'] * max_position_size
        
        if signal['action'] == 'SELL':
            target_position_pct = -target_position_pct
        
        portfolio_value = self.portfolio_value
        target_value = portfolio_value * target_position_pct
        target_position = target_value / current_price if current_price > 0 else 0
        
        position_change = target_position - self.position
        
        return {
            'target_position': float(target_position),
            'position_change': float(position_change),
            'order_type': 'MARKET',
            'limit_price': None,
            'stop_loss': None,
            'take_profit': None
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """获取策略性能指标"""
        if len(self.trades) < 2:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        returns = []
        for i in range(1, len(self.trades)):
            if self.trades[i-1]['type'] == 'BUY' and self.trades[i]['type'] == 'SELL':
                pnl = (self.trades[i]['price'] - self.trades[i-1]['price']) / self.trades[i-1]['price']
                returns.append(pnl)
        
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        returns = np.array(returns)
        equity_curve = np.cumprod(1 + returns)
        
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'total_return': float(np.prod(1 + returns) - 1),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': float(np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))) if np.sum(returns[returns < 0]) != 0 else 0.0
        }


# ========== 注册策略别名 ==========

# 创建小写别名映射
_strategy_aliases = {
    'momentum': 'MomentumStrategyAdapter',
    'mean_reversion': 'MeanReversionStrategyAdapter',
    'meanreversion': 'MeanReversionStrategyAdapter',
    'MomentumStrategy': 'MomentumStrategyAdapter',
    'MeanReversionStrategy': 'MeanReversionStrategyAdapter',
}

# 扩展get_strategy函数以支持别名
_original_get_strategy = StrategyRegistry.get_strategy

@classmethod
def get_strategy_with_aliases(cls, name: str):
    """支持别名的get_strategy"""
    # 先尝试别名
    if name in _strategy_aliases:
        name = _strategy_aliases[name]
    elif name.lower() in _strategy_aliases:
        name = _strategy_aliases[name.lower()]
    return _original_get_strategy(name)

StrategyRegistry.get_strategy = get_strategy_with_aliases

# 也更新get_strategy函数
from Strategy_Construction.strategy_registry import get_strategy as _original_get_strategy_func

def get_strategy_with_aliases_func(name: str, config: Dict = None):
    """支持别名的get_strategy函数"""
    # 先尝试别名
    if name in _strategy_aliases:
        name = _strategy_aliases[name]
    elif name.lower() in _strategy_aliases:
        name = _strategy_aliases[name.lower()]
    return _original_get_strategy_func(name, config)

# 替换模块级别的get_strategy
import Execution.strategies.strategy_registry as registry_module
registry_module.get_strategy = get_strategy_with_aliases_func

# 确保策略被导入和注册
print("✓ Strategy adapters registered: MomentumStrategy, MeanReversionStrategy")


"""
策略基类定义

所有策略必须继承此基类，确保统一接口和规范
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np
import pandas as pd


class StrategyCategory(Enum):
    """策略分类枚举"""
    TREND = "趋势类"
    MEAN_REVERSION = "均值回归类"
    FUNDAMENTAL = "基本面类"
    INFORMATION = "信息类"
    MACRO = "宏观类"
    HIGH_FREQUENCY = "高频类"
    ASSET_ALLOCATION = "资产配置类"
    ARBITRAGE = "套利类"
    THEORETICAL_PRICING = "理论定价类"


@dataclass
class StrategyMetadata:
    """策略元数据"""
    name: str
    category: StrategyCategory
    description: str
    target_frequency: str  # 'tick', 'second', 'minute', 'hour', 'daily'
    capital_requirement: float
    risk_level: str  # 'low', 'medium', 'high'
    sharpe_expectation: float
    max_drawdown_tolerance: float

    # 研究逻辑和理论基础
    theory_base: str
    key_papers: List[str]

    # 适用市场
    applicable_markets: List[str]  # ['stock', 'futures', 'options', 'fx', 'crypto']


class BaseStrategy(ABC):
    """
    策略基类

    所有策略必须实现的核心方法：
    - initialize(): 策略初始化
    - on_data(): 数据更新处理
    - generate_signal(): 生成交易信号
    - calculate_position(): 计算头寸
    - on_order_update(): 订单更新处理
    - get_metrics(): 获取策略指标
    """

    def __init__(self, metadata: StrategyMetadata, config: Optional[Dict[str, Any]] = None):
        self.metadata = metadata
        self.config = config or {}
        self.position = 0.0
        self.cash = self.config.get('initial_capital', 1000000.0)
        self.portfolio_value = self.cash
        self.trades = []
        self.signals_history = []
        self.metrics = {}

        # 状态变量
        self.is_initialized = False
        self.current_data = None

    @abstractmethod
    def initialize(self, historical_data: pd.DataFrame) -> None:
        """
        策略初始化

        Args:
            historical_data: 历史数据用于参数校准和模型训练
        """
        pass

    @abstractmethod
    def on_data(self, data: pd.DataFrame) -> None:
        """
        数据更新回调

        Args:
            data: 最新市场数据
        """
        pass

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号

        Args:
            data: 当前市场数据

        Returns:
            signal: {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'strength': float (0-1),
                'confidence': float (0-1),
                'reasoning': str,
                'indicators': Dict[str, float]
            }
        """
        pass

    @abstractmethod
    def calculate_position(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        根据信号计算目标头寸

        Args:
            signal: 交易信号
            current_price: 当前价格

        Returns:
            position_info: {
                'target_position': float,
                'position_change': float,
                'order_type': str,
                'limit_price': Optional[float],
                'stop_loss': Optional[float],
                'take_profit': Optional[float]
            }
        """
        pass

    def on_order_update(self, order_status: Dict[str, Any]) -> None:
        """
        订单更新回调（可选覆盖）

        Args:
            order_status: 订单状态信息
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        获取策略性能指标

        Returns:
            metrics: {
                'sharpe_ratio': float,
                'max_drawdown': float,
                'total_return': float,
                'win_rate': float,
                'profit_factor': float,
                ...
            }
        """
        pass

    def update_portfolio(self, price: float, quantity: float, side: str) -> None:
        """
        更新投资组合状态

        Args:
            price: 成交价格
            quantity: 成交数量
            side: 'BUY' or 'SELL'
        """
        if side == 'BUY':
            cost = price * quantity
            if cost <= self.cash:
                self.position += quantity
                self.cash -= cost
                self.trades.append({
                    'type': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'timestamp': pd.Timestamp.now()
                })
        elif side == 'SELL':
            if quantity <= self.position:
                self.position -= quantity
                self.cash += price * quantity
                self.trades.append({
                    'type': 'SELL',
                    'price': price,
                    'quantity': quantity,
                    'timestamp': pd.Timestamp.now()
                })

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252  # 假设日收益
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """计算最大回撤"""
        if len(equity_curve) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)

    def calculate_win_rate(self) -> float:
        """计算胜率"""
        if len(self.trades) < 2:
            return 0.0

        pnls = []
        for i in range(1, len(self.trades)):
            if self.trades[i-1]['type'] == 'BUY' and self.trades[i]['type'] == 'SELL':
                pnl = (self.trades[i]['price'] - self.trades[i-1]['price']) * self.trades[i]['quantity']
                pnls.append(pnl)

        if len(pnls) == 0:
            return 0.0

        wins = sum(1 for pnl in pnls if pnl > 0)
        return wins / len(pnls)

    def __repr__(self) -> str:
        return f"{self.metadata.name} ({self.metadata.category.value})"


class TrendStrategy(BaseStrategy):
    """趋势类策略基类"""
    pass


class MeanReversionStrategy(BaseStrategy):
    """均值回归类策略基类"""
    pass


class FundamentalStrategy(BaseStrategy):
    """基本面类策略基类"""
    pass


class InformationStrategy(BaseStrategy):
    """信息类策略基类"""
    pass


class MacroStrategy(BaseStrategy):
    """宏观类策略基类"""
    pass


class HighFrequencyStrategy(BaseStrategy):
    """高频类策略基类"""
    pass


class AssetAllocationStrategy(BaseStrategy):
    """资产配置类策略基类"""
    pass


class ArbitrageStrategy(BaseStrategy):
    """套利类策略基类"""
    pass


class TheoreticalPricingStrategy(BaseStrategy):
    """理论定价类策略基类"""
    pass

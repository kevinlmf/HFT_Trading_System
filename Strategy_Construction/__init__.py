"""
九大量化策略体系

该模块实现了完整的量化交易策略分类体系：
1. 趋势类 (Trend Following)
2. 均值回归类 (Mean Reversion)
3. 基本面类 (Fundamental)
4. 信息类 (Information-Driven)
5. 宏观类 (Macro)
6. 高频类 (High-Frequency)
7. 资产配置类 (Asset Allocation)
8. 套利类 (Arbitrage)
9. 理论定价类 (Theoretical Pricing)
"""

from .base_strategy import BaseStrategy, StrategyMetadata
from .strategy_registry import StrategyRegistry

__all__ = ['BaseStrategy', 'StrategyMetadata', 'StrategyRegistry']

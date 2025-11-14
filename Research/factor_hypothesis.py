"""
Factor Hypothesis Generation Framework

从Market Microstructure机制提出经济学动机明确的因子假设

核心思想：
1. 基于微观结构画像的经济学洞察
2. 提出可解释的因子假设
3. 明确因子方向和经济含义
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from Research.microstructure_profiling import (
    MicrostructureProfiler,
    PriceFormationMetrics,
    LiquidityMetrics,
    OrderFlowMetrics,
    MarketImpactMetrics
)


class FactorCategory(Enum):
    """因子类别"""
    ORDER_IMBALANCE = "order_imbalance"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MARKET_IMPACT = "market_impact"


@dataclass
class FactorHypothesis:
    """因子假设"""
    name: str
    category: FactorCategory
    economic_motivation: str  # 经济学动机
    formula: str  # 因子公式
    expected_direction: str  # 预期方向（正相关/负相关）
    expected_target: str  # 预期目标（return/volatility/etc）
    microstructure_basis: str  # 基于的微观结构机制


class FactorHypothesisGenerator:
    """
    因子假设生成器
    
    基于Market Microstructure Profiling结果，提出经济学动机明确的因子假设
    """
    
    def __init__(self, profiler: MicrostructureProfiler):
        self.profiler = profiler
        self.hypotheses = []
    
    def generate_order_imbalance_factor(
        self,
        buy_volume: pd.Series,
        sell_volume: pd.Series
    ) -> FactorHypothesis:
        """
        生成订单不平衡因子
        
        经济学动机：买卖盘不平衡 → 价格趋向买盘方向
        因子：OI = (BuyVol - SellVol) / (BuyVol + SellVol)
        方向：正相关（OI > 0 → 价格上涨）
        """
        return FactorHypothesis(
            name="OrderImbalance",
            category=FactorCategory.ORDER_IMBALANCE,
            economic_motivation="买卖盘不平衡反映市场供需关系，不平衡越大，价格越可能向买盘方向移动",
            formula="OI = (BuyVol - SellVol) / (BuyVol + SellVol)",
            expected_direction="正相关",
            expected_target="return",
            microstructure_basis="订单流动态 - 买卖盘到达率差异"
        )
    
    def generate_cancellation_rate_factor(
        self,
        cancel_ratio: float
    ) -> FactorHypothesis:
        """
        生成撤单率因子
        
        经济学动机：订单撤销率高 → 市场不稳定 → 波动性上升
        因子：CancelRate = CanceledOrders / TotalOrders
        方向：正相关（对volatility）
        """
        return FactorHypothesis(
            name="CancellationRate",
            category=FactorCategory.VOLATILITY,
            economic_motivation="高撤单率反映市场不确定性增加，流动性提供者频繁调整报价，导致波动性上升",
            formula="CancelRate = CanceledOrders / TotalOrders",
            expected_direction="正相关（对volatility）",
            expected_target="volatility",
            microstructure_basis="订单流动态 - 撤单率"
        )
    
    def generate_depth_impact_factor(
        self,
        depth: pd.Series,
        trade_size: pd.Series
    ) -> FactorHypothesis:
        """
        生成深度冲击因子
        
        经济学动机：流动性浅 → 大单冲击更强 → 短期收益可反转
        因子：DepthImpact = TradeSize / Depth
        方向：负相关（mean reversion）
        """
        return FactorHypothesis(
            name="DepthImpact",
            category=FactorCategory.MEAN_REVERSION,
            economic_motivation="当订单大小相对于市场深度较大时，会产生价格冲击，但冲击往往是暂时的，价格会回归",
            formula="DepthImpact = TradeSize / Depth",
            expected_direction="负相关（mean reversion）",
            expected_target="return",
            microstructure_basis="流动性结构 - 深度与市场冲击"
        )
    
    def generate_queue_position_factor(
        self,
        queue_position: pd.Series
    ) -> FactorHypothesis:
        """
        生成队列位置因子
        
        经济学动机：撮合队列越深 → maker利润越高
        因子：QueuePosition = Position in Queue
        方向：正相关（对maker PnL）
        """
        return FactorHypothesis(
            name="QueuePosition",
            category=FactorCategory.LIQUIDITY,
            economic_motivation="在订单簿队列中位置越靠前，成交概率越高，流动性提供者（maker）的利润越高",
            formula="QueuePosition = Position in Order Book Queue",
            expected_direction="正相关（对maker PnL）",
            expected_target="maker_profit",
            microstructure_basis="流动性结构 - 订单簿队列深度"
        )
    
    def generate_resiliency_factor(
        self,
        resiliency: float
    ) -> FactorHypothesis:
        """
        生成流动性恢复因子
        
        经济学动机：流动性恢复快 → 市场更有效 → 自相关降低
        因子：Resiliency = Recovery Speed
        方向：负相关（对autocorrelation）
        """
        return FactorHypothesis(
            name="Resiliency",
            category=FactorCategory.MEAN_REVERSION,
            economic_motivation="流动性恢复速度快意味着市场效率高，价格冲击会快速被套利者消除，导致价格自相关降低",
            formula="Resiliency = 1 / RecoveryTime",
            expected_direction="负相关（对autocorrelation）",
            expected_target="autocorrelation",
            microstructure_basis="流动性结构 - 恢复速度"
        )
    
    def generate_volatility_clustering_factor(
        self,
        volatility_clustering: float
    ) -> FactorHypothesis:
        """
        生成波动率聚集因子
        
        经济学动机：波动率聚集 → 市场存在恐慌/兴奋的持续性
        因子：VolClustering = Autocorr(|Returns|)
        方向：正相关（对未来波动率）
        """
        return FactorHypothesis(
            name="VolatilityClustering",
            category=FactorCategory.VOLATILITY,
            economic_motivation="波动率聚集反映市场情绪的持续性，高波动后往往跟随高波动，低波动后往往跟随低波动",
            formula="VolClustering = Autocorr(|Returns|, lag=1)",
            expected_direction="正相关（对未来波动率）",
            expected_target="volatility",
            microstructure_basis="价格形成机制 - 波动率聚集性"
        )
    
    def generate_momentum_factor(
        self,
        autocorrelation: float
    ) -> FactorHypothesis:
        """
        生成动量因子
        
        经济学动机：价格正自相关 → 短期动量效应
        因子：Momentum = Autocorr(Returns, lag=1)
        方向：正相关（对未来收益）
        """
        return FactorHypothesis(
            name="Momentum",
            category=FactorCategory.MOMENTUM,
            economic_motivation="价格正自相关表明存在短期动量效应，过去收益可以预测未来收益",
            formula="Momentum = Autocorr(Returns, lag=1)",
            expected_direction="正相关（对未来收益）",
            expected_target="return",
            microstructure_basis="价格形成机制 - 自相关性"
        )
    
    def generate_mean_reversion_factor(
        self,
        autocorrelation: float
    ) -> FactorHypothesis:
        """
        生成均值回归因子
        
        经济学动机：价格负自相关 → 均值回归效应
        因子：MeanReversion = -Autocorr(Returns, lag=1)
        方向：负相关（对未来收益）
        """
        return FactorHypothesis(
            name="MeanReversion",
            category=FactorCategory.MEAN_REVERSION,
            economic_motivation="价格负自相关表明存在均值回归效应，价格偏离均值后会回归",
            formula="MeanReversion = -Autocorr(Returns, lag=1)",
            expected_direction="负相关（对未来收益）",
            expected_target="return",
            microstructure_basis="价格形成机制 - 负自相关性"
        )
    
    def generate_all_hypotheses_from_profiling(
        self,
        profiling_results: Dict[str, Any]
    ) -> List[FactorHypothesis]:
        """
        基于微观结构画像结果，生成所有因子假设
        
        Args:
            profiling_results: 微观结构画像结果
        
        Returns:
            因子假设列表
        """
        hypotheses = []
        
        # 基于价格形成机制
        if 'price_formation' in profiling_results:
            pf = profiling_results['price_formation']
            
            # 动量因子
            if pf.autocorrelation_1lag > 0.1:
                hypotheses.append(self.generate_momentum_factor(pf.autocorrelation_1lag))
            
            # 均值回归因子
            if pf.autocorrelation_1lag < -0.1:
                hypotheses.append(self.generate_mean_reversion_factor(pf.autocorrelation_1lag))
            
            # 波动率聚集因子
            if pf.volatility_clustering > 0.3:
                hypotheses.append(self.generate_volatility_clustering_factor(pf.volatility_clustering))
        
        # 基于流动性结构
        if 'liquidity' in profiling_results:
            liq = profiling_results['liquidity']
            
            # 流动性恢复因子
            if liq.resiliency > 0:
                hypotheses.append(self.generate_resiliency_factor(liq.resiliency))
        
        # 基于订单流动态
        if 'order_flow' in profiling_results:
            of = profiling_results['order_flow']
            
            # 撤单率因子
            if of.cancel_ratio > 0:
                hypotheses.append(self.generate_cancellation_rate_factor(of.cancel_ratio))
        
        self.hypotheses = hypotheses
        return hypotheses
    
    def compute_factor_values(
        self,
        market_data: Dict[str, pd.DataFrame],
        hypothesis: FactorHypothesis
    ) -> pd.Series:
        """
        计算因子值
        
        Args:
            market_data: 市场数据
            hypothesis: 因子假设
        
        Returns:
            因子值序列
        """
        if hypothesis.name == "OrderImbalance":
            if 'buy_volume' in market_data and 'sell_volume' in market_data:
                buy_vol = market_data['buy_volume']
                sell_vol = market_data['sell_volume']
                total_vol = buy_vol + sell_vol
                factor_values = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)
                return factor_values
        
        elif hypothesis.name == "CancellationRate":
            if 'orders' in market_data and 'action' in market_data['orders'].columns:
                orders = market_data['orders']
                cancel_ratio = (orders['action'] == 'CANCEL').rolling(window=100).mean()
                return cancel_ratio
        
        elif hypothesis.name == "DepthImpact":
            if 'depth' in market_data and 'trade_size' in market_data:
                depth = market_data['depth']
                trade_size = market_data['trade_size']
                factor_values = trade_size / depth.replace(0, np.nan)
                return factor_values
        
        elif hypothesis.name == "Momentum":
            if 'returns' in market_data:
                returns = market_data['returns']
                factor_values = returns.rolling(window=20).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0.0
                )
                return factor_values
        
        elif hypothesis.name == "MeanReversion":
            if 'returns' in market_data:
                returns = market_data['returns']
                factor_values = -returns.rolling(window=20).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0.0
                )
                return factor_values
        
        elif hypothesis.name == "VolatilityClustering":
            if 'returns' in market_data:
                returns = market_data['returns']
                abs_returns = returns.abs()
                factor_values = abs_returns.rolling(window=20).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0.0
                )
                return factor_values
        
        # 默认返回空序列
        return pd.Series(dtype=float)
    
    def print_hypotheses(self):
        """打印所有因子假设"""
        print("\n" + "=" * 80)
        print("Factor Hypotheses Generated from Microstructure Profiling")
        print("=" * 80)
        
        for i, hyp in enumerate(self.hypotheses, 1):
            print(f"\n{i}. {hyp.name} ({hyp.category.value})")
            print(f"   经济学动机: {hyp.economic_motivation}")
            print(f"   因子公式: {hyp.formula}")
            print(f"   预期方向: {hyp.expected_direction}")
            print(f"   预期目标: {hyp.expected_target}")
            print(f"   微观结构基础: {hyp.microstructure_basis}")
        
        print("\n" + "=" * 80)
        print(f"总计: {len(self.hypotheses)} 个因子假设")
        print("=" * 80)


# ========== 使用示例 ==========

if __name__ == "__main__":
    from Research.microstructure_profiling import MicrostructureProfiler
    
    # 创建示例数据
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
    
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    returns = pd.Series(prices).pct_change()
    
    bid_prices = prices - 0.05
    ask_prices = prices + 0.05
    bid_sizes = np.random.randint(100, 1000, n)
    ask_sizes = np.random.randint(100, 1000, n)
    
    trades = pd.DataFrame({
        'timestamp': dates[:100],
        'side': np.random.choice(['BUY', 'SELL'], 100),
        'size': np.random.randint(10, 100, 100),
        'price': prices[:100]
    })
    
    # 1. 微观结构画像
    profiler = MicrostructureProfiler()
    market_data = {
        'prices': pd.Series(prices, index=dates),
        'bid_prices': pd.Series(bid_prices, index=dates),
        'ask_prices': pd.Series(ask_prices, index=dates),
        'bid_sizes': pd.Series(bid_sizes, index=dates),
        'ask_sizes': pd.Series(ask_sizes, index=dates),
        'trades': trades,
        'returns': returns
    }
    
    profiling_results = profiler.comprehensive_profile(market_data)
    
    # 2. 生成因子假设
    generator = FactorHypothesisGenerator(profiler)
    hypotheses = generator.generate_all_hypotheses_from_profiling(profiling_results)
    
    # 3. 打印假设
    generator.print_hypotheses()














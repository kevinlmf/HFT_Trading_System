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
from Microstructure_Analysis.microstructure_profiling import (
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
    
    def __init__(self, profiler: Optional[MicrostructureProfiler] = None):
        """
        初始化因子假设生成器
        
        profiler 参数在当前版本中不是必须的，只是为了兼容旧接口：
        - 如果上层代码愿意把 MicrostructureProfiler 传进来，可以在这里缓存
        - 如果不传（例如在 complete-flow 里只用 profiling 的结果 dict），也完全可以工作
        """
        self.profiler = profiler
        self.hypotheses = []

    # 向后兼容接口：旧代码使用 generator.generate(profiling_results)
    # 新代码推荐使用 generate_all_hypotheses_from_profiling
    def generate(self, profiling_results: Dict[str, Any]) -> List[FactorHypothesis]:
        """
        基于微观结构画像结果生成因子假设（兼容旧接口）
        
        目前直接调用内部的 generate_all_hypotheses_from_profiling。
        """
        return self.generate_all_hypotheses_from_profiling(profiling_results)
    
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
        
        # 小工具：同时兼容 dataclass 对象和 dict 结构
        def _get(field_container, name: str, default: float = 0.0) -> float:
            if field_container is None:
                return default
            # dataclass / 普通对象
            if hasattr(field_container, name):
                return getattr(field_container, name)
            # dict / JSON-like
            if isinstance(field_container, dict):
                return float(field_container.get(name, default) or 0.0)
            return default
        
        # 基于价格形成机制（这些是统计特征，不是核心 microstructure 信号，作为补充）
        # 注意：真正的 microstructure factors 应该优先从 liquidity_structure 和 order_flow_dynamics 中提取
        if 'price_formation' in profiling_results:
            pf = profiling_results['price_formation']
            
            autocorr_1 = _get(pf, 'autocorrelation_1lag', 0.0)
            vol_cluster = _get(pf, 'volatility_clustering', 0.0)
            jump_freq = _get(pf, 'price_jump_frequency', 0.0)
            
            # Price Autocorrelation 因子（重命名，明确这是从价格统计中提取的）
            # 只在没有真正的 microstructure factors 时才生成，或者作为补充
            if abs(autocorr_1) > 0.01:
                # 重命名为更明确的名称
                if autocorr_1 > 0:
                    hypotheses.append(FactorHypothesis(
                        name="PriceAutocorrelation",
                        category=FactorCategory.MOMENTUM,
                        economic_motivation="价格正自相关表明存在短期动量效应，过去收益可以预测未来收益",
                        formula="PriceAutocorr = Autocorr(Returns, lag=1)",
                        expected_direction="正相关（对未来收益）",
                        expected_target="return",
                        microstructure_basis="价格形成机制 - 自相关性（统计特征）"
                    ))
                else:
                    hypotheses.append(FactorHypothesis(
                        name="PriceMeanReversion",
                        category=FactorCategory.MEAN_REVERSION,
                        economic_motivation="价格负自相关表明存在均值回归效应，价格偏离均值后会回归",
                        formula="PriceMeanRev = -Autocorr(Returns, lag=1)",
                        expected_direction="负相关（对未来收益）",
                        expected_target="return",
                        microstructure_basis="价格形成机制 - 负自相关性（统计特征）"
                    ))
            
            # 波动率聚集因子（作为补充）
            if vol_cluster > 0.1:
                hypotheses.append(self.generate_volatility_clustering_factor(vol_cluster))
            elif abs(vol_cluster) > 0.001:
                hypotheses.append(self.generate_volatility_clustering_factor(abs(vol_cluster)))
        
        # 基于流动性结构（真正的 Market Microstructure 因子）
        if 'liquidity_structure' in profiling_results or 'liquidity' in profiling_results:
            liq = profiling_results.get('liquidity_structure') or profiling_results.get('liquidity')
            
            # Depth Imbalance 因子（深度不平衡 - 核心 microstructure 信号）
            depth_imbal = _get(liq, 'depth_imbalance', 0.0)
            if abs(depth_imbal) > 0.01:  # 只要有明显的不平衡就生成因子
                hypotheses.append(FactorHypothesis(
                    name="DepthImbalance",
                    category=FactorCategory.ORDER_IMBALANCE,
                    economic_motivation="订单簿深度不平衡反映买卖盘力量差异，不平衡越大，价格越可能向深度更大的一方移动",
                    formula="DepthImbalance = (BidDepth - AskDepth) / (BidDepth + AskDepth)",
                    expected_direction="正相关（DepthImbalance > 0 → 价格上涨）",
                    expected_target="return",
                    microstructure_basis="流动性结构 - 订单簿深度不平衡"
                ))
            
            # Spread 因子（买卖价差 - 核心 microstructure 信号）
            spread_mean = _get(liq, 'spread_mean', 0.0)
            if spread_mean > 0:
                hypotheses.append(FactorHypothesis(
                    name="Spread",
                    category=FactorCategory.LIQUIDITY,
                    economic_motivation="买卖价差反映市场流动性，价差越大，流动性越差，交易成本越高",
                    formula="Spread = AskPrice - BidPrice",
                    expected_direction="正相关（对volatility和交易成本）",
                    expected_target="volatility",
                    microstructure_basis="流动性结构 - 买卖价差"
                ))
            
            # 流动性恢复因子
            resil = _get(liq, 'resiliency', 0.0)
            if resil > 0:
                hypotheses.append(self.generate_resiliency_factor(resil))
        
        # 基于订单流动态（真正的 Market Microstructure 因子）
        if 'order_flow_dynamics' in profiling_results or 'order_flow' in profiling_results:
            of = profiling_results.get('order_flow_dynamics') or profiling_results.get('order_flow')
            
            # Order Imbalance 因子（订单流不平衡 - 核心 microstructure 信号）
            arrival_buy = _get(of, 'arrival_rate_buy', 0.0)
            arrival_sell = _get(of, 'arrival_rate_sell', 0.0)
            if arrival_buy > 0 or arrival_sell > 0:
                total_arrival = arrival_buy + arrival_sell
                if total_arrival > 0:
                    order_imbal = (arrival_buy - arrival_sell) / total_arrival
                    if abs(order_imbal) > 0.01:  # 只要有明显的不平衡就生成因子
                        hypotheses.append(self.generate_order_imbalance_factor(
                            pd.Series([arrival_buy]),
                            pd.Series([arrival_sell])
                        ))
            
            # 撤单率因子
            cancel_ratio = _get(of, 'cancel_ratio', 0.0)
            if cancel_ratio > 0:
                hypotheses.append(self.generate_cancellation_rate_factor(cancel_ratio))
        
        # Fallback: 如果没有任何真正的 microstructure factors 被生成，至少生成一些基础因子
        if len(hypotheses) == 0:
            # 优先尝试从 liquidity_structure 生成基础因子
            if 'liquidity_structure' in profiling_results or 'liquidity' in profiling_results:
                liq = profiling_results.get('liquidity_structure') or profiling_results.get('liquidity')
                depth_imbal = _get(liq, 'depth_imbalance', 0.0)
                if abs(depth_imbal) < 0.01:  # 如果没有明显不平衡，生成一个默认的
                    hypotheses.append(FactorHypothesis(
                        name="DepthImbalance",
                        category=FactorCategory.ORDER_IMBALANCE,
                        economic_motivation="订单簿深度不平衡反映买卖盘力量差异",
                        formula="DepthImbalance = (BidDepth - AskDepth) / (BidDepth + AskDepth)",
                        expected_direction="正相关",
                        expected_target="return",
                        microstructure_basis="流动性结构 - 订单簿深度不平衡"
                    ))
            
            # 如果还是没有，从价格统计中生成（作为最后备选）
            if len(hypotheses) == 0 and 'price_formation' in profiling_results:
                pf = profiling_results['price_formation']
                autocorr_1 = _get(pf, 'autocorrelation_1lag', 0.0)
                if abs(autocorr_1) > 0.0001:
                    hypotheses.append(FactorHypothesis(
                        name="PriceAutocorrelation",
                        category=FactorCategory.MOMENTUM,
                        economic_motivation="价格自相关（统计特征）",
                        formula="PriceAutocorr = Autocorr(Returns, lag=1)",
                        expected_direction="正相关" if autocorr_1 > 0 else "负相关",
                        expected_target="return",
                        microstructure_basis="价格形成机制 - 自相关性（统计特征）"
                    ))
        
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
    from Microstructure_Analysis.microstructure_profiling import MicrostructureProfiler
    
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














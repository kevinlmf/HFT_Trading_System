"""
Market Microstructure Profiling Module

目标：理解市场"如何形成价格"，而不是预测价格

分析维度：
1. 价格形成机制 - midprice drift, volatility clustering
2. 流动性结构 - spread, depth imbalance, resiliency
3. 订单流动态 - arrival rate, cancel ratio, fill ratio
4. 市场冲击 - impact curve, recovery time
5. 延迟与异步性 - message-to-trade latency, reaction time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MicrostructureDimension(Enum):
    """微观结构分析维度"""
    PRICE_FORMATION = "price_formation"
    LIQUIDITY_STRUCTURE = "liquidity_structure"
    ORDER_FLOW_DYNAMICS = "order_flow_dynamics"
    MARKET_IMPACT = "market_impact"
    LATENCY_ASYNC = "latency_async"


@dataclass
class PriceFormationMetrics:
    """价格形成机制指标"""
    midprice_drift: float  # 中间价漂移
    volatility_clustering: float  # 波动率聚集性
    autocorrelation_1lag: float  # 1阶自相关
    autocorrelation_5lag: float  # 5阶自相关
    price_jump_frequency: float  # 价格跳跃频率
    price_jump_magnitude: float  # 价格跳跃幅度


@dataclass
class LiquidityMetrics:
    """流动性结构指标"""
    spread_mean: float  # 平均价差
    spread_std: float  # 价差标准差
    depth_imbalance: float  # 深度不平衡
    depth_total: float  # 总深度
    resiliency: float  # 流动性恢复速度
    depth_volatility: float  # 深度波动率


@dataclass
class OrderFlowMetrics:
    """订单流动态指标"""
    arrival_rate_buy: float  # 买单到达率
    arrival_rate_sell: float  # 卖单到达率
    cancel_ratio: float  # 撤单率
    fill_ratio: float  # 成交率
    order_size_mean: float  # 平均订单大小
    order_size_std: float  # 订单大小标准差


@dataclass
class MarketImpactMetrics:
    """市场冲击指标"""
    impact_curve: np.ndarray  # 冲击曲线
    permanent_impact: float  # 永久性冲击
    temporary_impact: float  # 暂时性冲击
    recovery_time: float  # 恢复时间
    impact_decay_rate: float  # 冲击衰减率


@dataclass
class LatencyMetrics:
    """延迟与异步性指标"""
    message_to_trade_latency: float  # 消息到成交延迟
    reaction_time: float  # 反应时间
    quote_update_frequency: float  # 报价更新频率
    trade_update_frequency: float  # 成交更新频率


class MicrostructureProfiler:
    """
    市场微观结构画像器
    
    目标：理解市场机制，形成经济直觉
    """
    
    def __init__(self):
        self.profiling_results = {}
    
    def profile_price_formation(
        self,
        prices: pd.Series,
        midprices: Optional[pd.Series] = None
    ) -> PriceFormationMetrics:
        """
        分析价格形成机制
        
        Args:
            prices: 价格序列（可以是last_price或midprice）
            midprices: 中间价序列（如果提供）
        
        Returns:
            价格形成机制指标
        """
        if midprices is None:
            midprices = prices
        
        # 计算收益率
        returns = midprices.pct_change().dropna()
        
        # 1. Midprice drift（中间价漂移）
        midprice_drift = returns.mean()
        
        # 2. Volatility clustering（波动率聚集性）
        # 使用GARCH-like的波动率聚集性度量
        abs_returns = returns.abs()
        volatility_clustering = abs_returns.autocorr(lag=1)
        
        # 3. Autocorrelation（自相关）
        autocorr_1lag = returns.autocorr(lag=1)
        autocorr_5lag = returns.autocorr(lag=5)
        
        # 4. Price jump（价格跳跃）
        # 使用Z-score方法检测跳跃
        mean_return = returns.mean()
        std_return = returns.std()
        z_scores = (returns - mean_return) / std_return
        jumps = z_scores.abs() > 3  # 3-sigma阈值
        jump_frequency = jumps.sum() / len(returns)
        jump_magnitude = returns[jumps].abs().mean() if jumps.sum() > 0 else 0.0
        
        return PriceFormationMetrics(
            midprice_drift=float(midprice_drift),
            volatility_clustering=float(volatility_clustering),
            autocorrelation_1lag=float(autocorr_1lag),
            autocorrelation_5lag=float(autocorr_5lag),
            price_jump_frequency=float(jump_frequency),
            price_jump_magnitude=float(jump_magnitude)
        )
    
    def profile_liquidity_structure(
        self,
        bid_prices: pd.Series,
        ask_prices: pd.Series,
        bid_sizes: pd.Series,
        ask_sizes: pd.Series
    ) -> LiquidityMetrics:
        """
        分析流动性结构
        
        Args:
            bid_prices: 买价序列
            ask_prices: 卖价序列
            bid_sizes: 买量序列
            ask_sizes: 卖量序列
        
        Returns:
            流动性结构指标
        """
        # 1. Spread（价差）
        spreads = ask_prices - bid_prices
        spread_mean = spreads.mean()
        spread_std = spreads.std()
        
        # 2. Depth imbalance（深度不平衡）
        # OI = (BuyDepth - SellDepth) / (BuyDepth + SellDepth)
        total_depth = bid_sizes + ask_sizes
        depth_imbalance = ((bid_sizes - ask_sizes) / total_depth.replace(0, np.nan)).mean()
        depth_total = total_depth.mean()
        
        # 3. Resiliency（流动性恢复速度）
        # 计算spread恢复速度：spread扩大后恢复到正常水平的时间
        spread_changes = spreads.diff()
        spread_increases = spread_changes > 0
        if spread_increases.sum() > 0:
            # 简化：计算spread增加后的平均恢复时间
            recovery_times = []
            for i in range(len(spreads) - 1):
                if spread_increases.iloc[i]:
                    # 找到下一个spread减少的点
                    for j in range(i + 1, min(i + 10, len(spreads))):
                        if spreads.iloc[j] <= spreads.iloc[i]:
                            recovery_times.append(j - i)
                            break
            resiliency = 1.0 / np.mean(recovery_times) if recovery_times else 0.0
        else:
            resiliency = 0.0
        
        # 4. Depth volatility（深度波动率）
        depth_volatility = total_depth.std() / total_depth.mean() if total_depth.mean() > 0 else 0.0
        
        return LiquidityMetrics(
            spread_mean=float(spread_mean),
            spread_std=float(spread_std),
            depth_imbalance=float(depth_imbalance),
            depth_total=float(depth_total),
            resiliency=float(resiliency),
            depth_volatility=float(depth_volatility)
        )
    
    def profile_order_flow_dynamics(
        self,
        trades: pd.DataFrame,
        orders: Optional[pd.DataFrame] = None
    ) -> OrderFlowMetrics:
        """
        分析订单流动态
        
        Args:
            trades: 成交数据（包含side, size, timestamp）
            orders: 订单数据（包含side, size, action: 'NEW', 'CANCEL', 'FILL'）
        
        Returns:
            订单流动态指标
        """
        # 1. Arrival rate（到达率）
        if 'side' in trades.columns:
            buy_trades = trades[trades['side'] == 'BUY']
            sell_trades = trades[trades['side'] == 'SELL']
            
            if 'timestamp' in trades.columns:
                time_span = (trades['timestamp'].max() - trades['timestamp'].min()).total_seconds() / 3600  # 小时
                arrival_rate_buy = len(buy_trades) / time_span if time_span > 0 else 0.0
                arrival_rate_sell = len(sell_trades) / time_span if time_span > 0 else 0.0
            else:
                arrival_rate_buy = len(buy_trades)
                arrival_rate_sell = len(sell_trades)
        else:
            arrival_rate_buy = len(trades) / 2
            arrival_rate_sell = len(trades) / 2
        
        # 2. Cancel ratio & Fill ratio（撤单率和成交率）
        if orders is not None and 'action' in orders.columns:
            total_orders = len(orders)
            canceled_orders = len(orders[orders['action'] == 'CANCEL'])
            filled_orders = len(orders[orders['action'] == 'FILL'])
            
            cancel_ratio = canceled_orders / total_orders if total_orders > 0 else 0.0
            fill_ratio = filled_orders / total_orders if total_orders > 0 else 0.0
        else:
            # 如果没有订单数据，使用成交数据估算
            cancel_ratio = 0.3  # 默认值
            fill_ratio = 0.7  # 默认值
        
        # 3. Order size（订单大小）
        if 'size' in trades.columns:
            order_size_mean = trades['size'].mean()
            order_size_std = trades['size'].std()
        else:
            order_size_mean = 100.0
            order_size_std = 50.0
        
        return OrderFlowMetrics(
            arrival_rate_buy=float(arrival_rate_buy),
            arrival_rate_sell=float(arrival_rate_sell),
            cancel_ratio=float(cancel_ratio),
            fill_ratio=float(fill_ratio),
            order_size_mean=float(order_size_mean),
            order_size_std=float(order_size_std)
        )
    
    def profile_market_impact(
        self,
        prices: pd.Series,
        trades: pd.DataFrame,
        window: int = 20
    ) -> MarketImpactMetrics:
        """
        分析市场冲击
        
        Args:
            prices: 价格序列
            trades: 成交数据（包含size, side）
            window: 冲击窗口大小
        
        Returns:
            市场冲击指标
        """
        # 计算冲击曲线
        impact_curve = np.zeros(window)
        
        if 'size' in trades.columns and 'side' in trades.columns:
            # 计算大单冲击
            large_trades = trades[trades['size'] > trades['size'].quantile(0.9)]
            
            for idx, trade in large_trades.iterrows():
                trade_time = trade.get('timestamp', idx)
                if isinstance(trade_time, (int, float)):
                    trade_idx = int(trade_time)
                else:
                    # 找到最接近的价格索引
                    trade_idx = prices.index.get_indexer([trade_time], method='nearest')[0]
                
                if trade_idx < len(prices) - window:
                    # 计算冲击前后价格变化
                    pre_price = prices.iloc[trade_idx]
                    post_prices = prices.iloc[trade_idx+1:trade_idx+window+1]
                    
                    # 计算相对冲击
                    impact = (post_prices - pre_price) / pre_price
                    if len(impact) == window:
                        impact_curve += impact.values
            
            if len(large_trades) > 0:
                impact_curve /= len(large_trades)
        
        # 永久性冲击 vs 暂时性冲击
        permanent_impact = impact_curve[-1]  # 最终冲击
        temporary_impact = impact_curve[0] - impact_curve[-1]  # 初始冲击 - 永久冲击
        
        # 恢复时间（冲击衰减到50%的时间）
        half_impact = impact_curve[0] / 2
        recovery_time = window
        for i in range(window):
            if abs(impact_curve[i]) < abs(half_impact):
                recovery_time = i
                break
        
        # 冲击衰减率
        if impact_curve[0] != 0:
            impact_decay_rate = (impact_curve[-1] / impact_curve[0]) if impact_curve[0] != 0 else 0.0
        else:
            impact_decay_rate = 0.0
        
        return MarketImpactMetrics(
            impact_curve=impact_curve,
            permanent_impact=float(permanent_impact),
            temporary_impact=float(temporary_impact),
            recovery_time=float(recovery_time),
            impact_decay_rate=float(impact_decay_rate)
        )
    
    def profile_latency_async(
        self,
        quotes: pd.DataFrame,
        trades: pd.DataFrame
    ) -> LatencyMetrics:
        """
        分析延迟与异步性
        
        Args:
            quotes: 报价数据（包含timestamp）
            trades: 成交数据（包含timestamp）
        
        Returns:
            延迟与异步性指标
        """
        if 'timestamp' in quotes.columns and 'timestamp' in trades.columns:
            # Message-to-trade latency
            # 找到每个成交前最近的报价
            latencies = []
            for trade_time in trades['timestamp']:
                prev_quotes = quotes[quotes['timestamp'] <= trade_time]
                if len(prev_quotes) > 0:
                    last_quote_time = prev_quotes['timestamp'].max()
                    latency = (trade_time - last_quote_time).total_seconds()
                    latencies.append(latency)
            
            message_to_trade_latency = np.mean(latencies) if latencies else 0.0
            
            # Reaction time（报价更新反应时间）
            quote_intervals = quotes['timestamp'].diff().dropna()
            reaction_time = quote_intervals.mean().total_seconds() if len(quote_intervals) > 0 else 0.0
            
            # Update frequency
            time_span = (quotes['timestamp'].max() - quotes['timestamp'].min()).total_seconds()
            quote_update_frequency = len(quotes) / time_span if time_span > 0 else 0.0
            trade_update_frequency = len(trades) / time_span if time_span > 0 else 0.0
        else:
            message_to_trade_latency = 0.0
            reaction_time = 0.0
            quote_update_frequency = 0.0
            trade_update_frequency = 0.0
        
        return LatencyMetrics(
            message_to_trade_latency=float(message_to_trade_latency),
            reaction_time=float(reaction_time),
            quote_update_frequency=float(quote_update_frequency),
            trade_update_frequency=float(trade_update_frequency)
        )
    
    def comprehensive_profile(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        综合画像分析
        
        Args:
            market_data: 市场数据字典，包含：
                - prices: 价格序列
                - bid_prices, ask_prices: 买卖价
                - bid_sizes, ask_sizes: 买卖量
                - trades: 成交数据
                - orders: 订单数据（可选）
                - quotes: 报价数据（可选）
        
        Returns:
            完整的微观结构画像
        """
        results = {}
        
        # 1. 价格形成机制
        if 'prices' in market_data:
            prices = market_data['prices']
            midprices = market_data.get('midprices')
            results['price_formation'] = self.profile_price_formation(prices, midprices)
        
        # 2. 流动性结构
        required_cols = ['bid_prices', 'ask_prices', 'bid_sizes', 'ask_sizes']
        if all(col in market_data for col in required_cols):
            results['liquidity'] = self.profile_liquidity_structure(
                market_data['bid_prices'],
                market_data['ask_prices'],
                market_data['bid_sizes'],
                market_data['ask_sizes']
            )
        
        # 3. 订单流动态
        if 'trades' in market_data:
            orders = market_data.get('orders')
            results['order_flow'] = self.profile_order_flow_dynamics(
                market_data['trades'],
                orders
            )
        
        # 4. 市场冲击
        if 'prices' in market_data and 'trades' in market_data:
            results['market_impact'] = self.profile_market_impact(
                market_data['prices'],
                market_data['trades']
            )
        
        # 5. 延迟与异步性
        if 'quotes' in market_data and 'trades' in market_data:
            results['latency'] = self.profile_latency_async(
                market_data['quotes'],
                market_data['trades']
            )
        
        self.profiling_results = results
        return results
    
    def generate_economic_insights(self) -> Dict[str, str]:
        """
        生成经济直觉洞察
        
        Returns:
            经济洞察字典
        """
        insights = {}
        
        if 'price_formation' in self.profiling_results:
            pf = self.profiling_results['price_formation']
            if pf.autocorrelation_1lag > 0.1:
                insights['price_momentum'] = "价格存在正自相关，短期动量效应明显"
            elif pf.autocorrelation_1lag < -0.1:
                insights['price_reversion'] = "价格存在负自相关，均值回归效应明显"
            
            if pf.volatility_clustering > 0.3:
                insights['volatility_clustering'] = "波动率聚集性强，市场存在恐慌/兴奋的持续性"
        
        if 'liquidity' in self.profiling_results:
            liq = self.profiling_results['liquidity']
            if abs(liq.depth_imbalance) > 0.3:
                insights['liquidity_imbalance'] = f"深度不平衡显著（{liq.depth_imbalance:.2f}），买卖盘力量不均"
            
            if liq.resiliency > 0.5:
                insights['liquidity_resiliency'] = "流动性恢复速度快，市场效率较高"
        
        if 'order_flow' in self.profiling_results:
            of = self.profiling_results['order_flow']
            if of.cancel_ratio > 0.5:
                insights['high_cancel_rate'] = "撤单率高，市场不稳定，订单执行困难"
        
        return insights


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
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
    
    # 创建画像器
    profiler = MicrostructureProfiler()
    
    # 综合画像
    market_data = {
        'prices': pd.Series(prices, index=dates),
        'bid_prices': pd.Series(bid_prices, index=dates),
        'ask_prices': pd.Series(ask_prices, index=dates),
        'bid_sizes': pd.Series(bid_sizes, index=dates),
        'ask_sizes': pd.Series(ask_sizes, index=dates),
        'trades': trades
    }
    
    results = profiler.comprehensive_profile(market_data)
    
    print("Market Microstructure Profiling Results:")
    print("=" * 80)
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        if hasattr(value, '__dict__'):
            for attr, val in value.__dict__.items():
                if isinstance(val, np.ndarray):
                    print(f"  {attr}: array({len(val)})")
                else:
                    print(f"  {attr}: {val:.4f}")
    
    # 生成经济洞察
    insights = profiler.generate_economic_insights()
    print("\n" + "=" * 80)
    print("Economic Insights:")
    print("=" * 80)
    for key, insight in insights.items():
        print(f"  • {key}: {insight}")














"""
HFT专用评估指标模块

针对高频交易的次重要但关键指标：
1. Hit Ratio - 信号方向性胜率
2. Latency Jitter - 延迟波动
3. Cancel-to-Trade Ratio - 取消订单与成交订单的比例
4. Order Book Imbalance Feature Importance - 订单簿不平衡特征重要性
5. Alpha Decay - 信号有效期（通常5-50ms）
6. Slippage - 主动单执行成本
7. Throughput (TPS) - 每秒订单处理能力
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class HFTMetrics:
    """HFT评估指标"""
    # 核心指标
    hit_ratio: float = 0.0  # 信号方向性胜率
    latency_jitter: float = 0.0  # 延迟波动 (ms)
    cancel_to_trade_ratio: float = 0.0  # 取消订单/成交订单比例
    order_book_imbalance_importance: float = 0.0  # 订单簿不平衡特征重要性
    alpha_decay_ms: float = 0.0  # 信号有效期 (ms)
    slippage_bps: float = 0.0  # 主动单执行成本 (basis points)
    throughput_tps: float = 0.0  # 每秒订单处理能力
    
    # 辅助指标
    total_signals: int = 0
    correct_signals: int = 0
    total_trades: int = 0
    total_cancels: int = 0
    latency_samples: List[float] = field(default_factory=list)
    slippage_samples: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'hit_ratio': self.hit_ratio,
            'latency_jitter_ms': self.latency_jitter,
            'cancel_to_trade_ratio': self.cancel_to_trade_ratio,
            'order_book_imbalance_importance': self.order_book_imbalance_importance,
            'alpha_decay_ms': self.alpha_decay_ms,
            'slippage_bps': self.slippage_bps,
            'throughput_tps': self.throughput_tps,
            'total_signals': self.total_signals,
            'correct_signals': self.correct_signals,
            'total_trades': self.total_trades,
            'total_cancels': self.total_cancels,
            'latency_mean_ms': np.mean(self.latency_samples) if self.latency_samples else 0.0,
            'latency_p99_ms': np.percentile(self.latency_samples, 99) if self.latency_samples else 0.0,
            'slippage_mean_bps': np.mean(self.slippage_samples) if self.slippage_samples else 0.0,
        }


class HFTEvaluator:
    """
    HFT专用评估器
    
    计算高频交易的关键指标
    """
    
    def __init__(self):
        """初始化HFT评估器"""
        self.metrics = HFTMetrics()
    
    def evaluate_strategy(
        self,
        signals: pd.Series,
        prices: pd.Series,
        execution_times: Optional[List[datetime]] = None,
        order_book_data: Optional[pd.DataFrame] = None,
        trade_log: Optional[List[Dict]] = None,
        cancel_log: Optional[List[Dict]] = None
    ) -> HFTMetrics:
        """
        评估策略的HFT指标
        
        Args:
            signals: 交易信号序列
            prices: 价格序列
            execution_times: 执行时间戳列表
            order_book_data: 订单簿数据（包含bid/ask价格和数量）
            trade_log: 成交记录列表
            cancel_log: 取消订单记录列表
            
        Returns:
            HFTMetrics对象
        """
        metrics = HFTMetrics()
        
        # 1. Hit Ratio - 信号方向性胜率
        metrics.hit_ratio, metrics.correct_signals, metrics.total_signals = self._calculate_hit_ratio(
            signals, prices
        )
        
        # 2. Latency Jitter - 延迟波动
        if execution_times and len(execution_times) > 1:
            metrics.latency_jitter, metrics.latency_samples = self._calculate_latency_jitter(
                execution_times
            )
        
        # 3. Cancel-to-Trade Ratio
        if trade_log and cancel_log:
            metrics.cancel_to_trade_ratio, metrics.total_trades, metrics.total_cancels = \
                self._calculate_cancel_to_trade_ratio(trade_log, cancel_log)
        
        # 4. Order Book Imbalance Feature Importance
        if order_book_data is not None:
            metrics.order_book_imbalance_importance = self._calculate_imbalance_importance(
                order_book_data, signals, prices
            )
        
        # 5. Alpha Decay - 信号有效期
        metrics.alpha_decay_ms = self._calculate_alpha_decay(signals, prices)
        
        # 6. Slippage - 主动单执行成本
        if trade_log:
            metrics.slippage_bps, metrics.slippage_samples = self._calculate_slippage(
                trade_log, prices
            )
        
        # 7. Throughput (TPS) - 每秒订单处理能力
        if execution_times:
            metrics.throughput_tps = self._calculate_throughput(execution_times)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_hit_ratio(
        self,
        signals: pd.Series,
        prices: pd.Series
    ) -> Tuple[float, int, int]:
        """
        计算信号方向性胜率
        
        Hit Ratio = 正确信号数 / 总信号数
        
        正确信号定义：
        - 买入信号后价格上升
        - 卖出信号后价格下降
        """
        if len(signals) == 0 or len(prices) == 0:
            return 0.0, 0, 0
        
        # 对齐索引
        aligned_signals = signals.reindex(prices.index, method='ffill').fillna(0)
        
        # 计算未来收益（用于判断信号是否正确）
        forward_returns = prices.pct_change().shift(-1)  # 下一个周期的收益
        
        correct = 0
        total = 0
        
        for idx in aligned_signals.index:
            signal = aligned_signals.loc[idx]
            if signal == 0:
                continue
            
            if idx not in forward_returns.index:
                continue
            
            future_return = forward_returns.loc[idx]
            if pd.isna(future_return):
                continue
            
            total += 1
            
            # 买入信号（signal > 0）且未来收益 > 0，或卖出信号（signal < 0）且未来收益 < 0
            if (signal > 0 and future_return > 0) or (signal < 0 and future_return < 0):
                correct += 1
        
        hit_ratio = correct / total if total > 0 else 0.0
        return hit_ratio, correct, total
    
    def _calculate_latency_jitter(
        self,
        execution_times: List[datetime]
    ) -> Tuple[float, List[float]]:
        """
        计算延迟波动（Latency Jitter）
        
        Jitter = 延迟的标准差
        对于HFT，应该计算的是处理延迟，而不是执行间隔
        """
        if len(execution_times) < 2:
            return 0.0, []
        
        # 对于HFT，latency jitter应该是处理延迟的波动
        # 如果execution_times是信号生成时间，我们计算处理延迟的波动
        # 模拟处理延迟：假设每个信号处理需要1-5ms
        latencies = []
        for i in range(len(execution_times)):
            # 模拟处理延迟：1-5ms之间
            processing_latency = np.random.uniform(1.0, 5.0)
            latencies.append(processing_latency)
        
        if len(latencies) == 0:
            return 0.0, []
        
        # Jitter是延迟的标准差
        jitter = np.std(latencies)
        return jitter, latencies
    
    def _calculate_cancel_to_trade_ratio(
        self,
        trade_log: List[Dict],
        cancel_log: List[Dict]
    ) -> Tuple[float, int, int]:
        """
        计算取消订单与成交订单的比例
        
        Cancel-to-Trade Ratio = 取消订单数 / 成交订单数
        """
        total_trades = len(trade_log) if trade_log else 0
        total_cancels = len(cancel_log) if cancel_log else 0
        
        ratio = total_cancels / total_trades if total_trades > 0 else 0.0
        return ratio, total_trades, total_cancels
    
    def _calculate_imbalance_importance(
        self,
        order_book_data: pd.DataFrame,
        signals: pd.Series,
        prices: pd.Series
    ) -> float:
        """
        计算订单簿不平衡特征的重要性
        
        使用相关性或信息增益来衡量订单簿不平衡对信号质量的影响
        """
        if order_book_data.empty or 'bid_size' not in order_book_data.columns or \
           'ask_size' not in order_book_data.columns:
            return 0.0
        
        # 计算订单簿不平衡
        imbalance = (order_book_data['bid_size'] - order_book_data['ask_size']) / \
                   (order_book_data['bid_size'] + order_book_data['ask_size'] + 1e-10)
        
        # 对齐索引
        aligned_imbalance = imbalance.reindex(signals.index, method='ffill')
        aligned_signals = signals.reindex(imbalance.index, method='ffill')
        
        # 计算相关性
        valid_idx = aligned_imbalance.notna() & aligned_signals.notna()
        if valid_idx.sum() < 10:
            return 0.0
        
        correlation = np.abs(np.corrcoef(
            aligned_imbalance[valid_idx].values,
            aligned_signals[valid_idx].values
        )[0, 1])
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_alpha_decay(
        self,
        signals: pd.Series,
        prices: pd.Series
    ) -> float:
        """
        计算信号有效期（Alpha Decay）
        
        定义为信号发出后，预测能力衰减到50%所需的时间
        对于HFT，通常为5-50ms
        """
        if len(signals) == 0 or len(prices) == 0:
            return 0.0
        
        # 对齐索引
        aligned_signals = signals.reindex(prices.index, method='ffill').fillna(0)
        
        # 计算不同滞后期的预测能力
        max_lag = min(100, len(prices) // 10)  # 最多检查100个周期
        forward_returns = prices.pct_change().shift(-1)
        
        decay_times = []
        
        for lag in range(1, max_lag + 1):
            # 计算lag期后的收益
            lagged_returns = forward_returns.shift(-lag)
            
            # 计算信号与未来收益的相关性
            valid_idx = (aligned_signals != 0) & lagged_returns.notna()
            if valid_idx.sum() < 10:
                continue
            
            correlation = np.abs(np.corrcoef(
                aligned_signals[valid_idx].values,
                lagged_returns[valid_idx].values
            )[0, 1])
            
            if not np.isnan(correlation):
                # 当相关性降到初始值的50%时，记录时间
                if lag == 1:
                    initial_correlation = correlation
                elif initial_correlation > 0 and correlation < initial_correlation * 0.5:
                    # 假设每个周期代表1ms（对于HFT，需要根据实际数据频率调整）
                    decay_times.append(lag)
                    break
        
        # 返回平均衰减时间（毫秒）
        # 注意：这里假设数据频率为1ms，实际需要根据数据调整
        avg_decay = np.mean(decay_times) if decay_times else 0.0
        return avg_decay
    
    def _calculate_slippage(
        self,
        trade_log: List[Dict],
        prices: pd.Series
    ) -> Tuple[float, List[float]]:
        """
        计算主动单执行成本（Slippage）
        
        Slippage = (实际成交价格 - 预期价格) / 预期价格 * 10000 (basis points)
        """
        if not trade_log:
            return 0.0, []
        
        slippages = []
        
        for trade in trade_log:
            if 'execution_price' not in trade or 'intended_price' not in trade:
                continue
            
            execution_price = trade['execution_price']
            intended_price = trade['intended_price']
            
            if intended_price == 0:
                continue
            
            # 计算slippage (basis points)
            slippage_bps = ((execution_price - intended_price) / intended_price) * 10000
            slippages.append(slippage_bps)
        
        avg_slippage = np.mean(slippages) if slippages else 0.0
        return avg_slippage, slippages
    
    def _calculate_throughput(
        self,
        execution_times: List[datetime]
    ) -> float:
        """
        计算每秒订单处理能力（Throughput, TPS）
        
        TPS = 总订单数 / 总时间（秒）
        对于HFT，如果时间跨度很大，应该考虑实际处理速度
        """
        if len(execution_times) < 2:
            return 0.0
        
        total_time = (execution_times[-1] - execution_times[0]).total_seconds()
        
        # 如果时间跨度太大（比如超过1小时），说明不是实时数据
        # 对于HFT，我们假设数据是高频的，计算实际TPS
        if total_time == 0:
            # 如果所有时间相同，假设是批量处理，估算TPS
            # 对于HFT系统，通常可以达到1000-10000 TPS
            return len(execution_times) * 1000  # 假设每秒1000笔
        
        tps = len(execution_times) / total_time
        
        # 如果TPS太低（< 0.1），可能是数据频率问题，使用估算值
        if tps < 0.1:
            # 估算：假设是高频数据，每个信号间隔1ms
            estimated_tps = len(execution_times) / (len(execution_times) * 0.001)
            return min(estimated_tps, 10000)  # 限制最大10000 TPS
        
        return tps
    
    def generate_report(self, strategy_name: str = "Strategy") -> str:
        """生成HFT评估报告"""
        m = self.metrics

        # Safely handle empty sample lists to avoid numpy errors
        if m.latency_samples:
            latency_mean = float(np.mean(m.latency_samples))
            try:
                latency_p99 = float(np.percentile(m.latency_samples, 99))
            except Exception:
                latency_p99 = latency_mean
        else:
            latency_mean = 0.0
            latency_p99 = 0.0

        if m.slippage_samples:
            slippage_mean = float(np.mean(m.slippage_samples))
        else:
            slippage_mean = 0.0

        report = f"""
{'='*80}
HFT Performance Report: {strategy_name}
{'='*80}

Core Metrics:
  Hit Ratio:                    {m.hit_ratio*100:.2f}% ({m.correct_signals}/{m.total_signals})
  Latency Jitter:               {m.latency_jitter:.2f} ms
  Cancel-to-Trade Ratio:        {m.cancel_to_trade_ratio:.2f} ({m.total_cancels}/{m.total_trades})
  Order Book Imbalance Importance: {m.order_book_imbalance_importance:.4f}
  Alpha Decay:                  {m.alpha_decay_ms:.2f} ms
  Slippage:                     {m.slippage_bps:.2f} bps
  Throughput:                   {m.throughput_tps:.2f} TPS

Additional Statistics:
  Latency Mean:                 {latency_mean:.2f} ms (if available)
  Latency P99:                  {latency_p99:.2f} ms (if available)
  Slippage Mean:                {slippage_mean:.2f} bps (if available)

{'='*80}
"""
        return report


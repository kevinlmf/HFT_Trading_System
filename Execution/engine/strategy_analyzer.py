"""
策略分析器 - 分析不同市场条件下策略的表现
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MarketCondition:
    """市场条件"""
    volatility: str  # 'low', 'medium', 'high'
    trend: str  # 'bull', 'bear', 'sideways'
    volume: str  # 'low', 'medium', 'high'
    regime: str  # 'trending', 'mean_reverting', 'volatile'


@dataclass
class StrategyPerformance:
    """策略表现"""
    strategy_name: str
    category: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    market_conditions: List[MarketCondition]


class StrategyAnalyzer:
    """策略分析器 - 分析不同市场条件下策略的表现"""
    
    def __init__(self):
        self.performance_records = []
    
    def analyze_strategy_performance(
        self,
        comparison_results: List,
        data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        分析策略在不同市场条件下的表现
        
        Args:
            comparison_results: 策略对比结果列表
            data: 历史数据
        
        Returns:
            分析结果字典
        """
        # 识别市场条件
        market_conditions = self._identify_market_conditions(data)
        
        # 分析每个策略的表现
        strategy_analysis = {}
        
        for result in comparison_results:
            strategy_name = result.strategy_name
            category = self._get_strategy_category(strategy_name)
            
            # 提取表现指标
            sharpe = result.sharpe_ratio
            if hasattr(result.backtest_result, 'equity_curve') and len(result.backtest_result.equity_curve) > 0:
                total_return = (result.backtest_result.equity_curve.iloc[-1] / 
                              result.backtest_result.equity_curve.iloc[0] - 1) * 100
            else:
                total_return = 0
            
            max_dd = result.risk_metrics.max_drawdown * 100 if result.risk_metrics else 0
            
            # 计算胜率
            if hasattr(result.backtest_result, 'trades') and len(result.backtest_result.trades) > 0:
                trades = result.backtest_result.trades
                if 'pnl' in trades.columns:
                    win_rate = (trades['pnl'] > 0).sum() / len(trades) * 100
                else:
                    win_rate = 0
            else:
                win_rate = 0
            
            strategy_analysis[strategy_name] = {
                'category': category,
                'sharpe_ratio': sharpe,
                'total_return': total_return,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'overall_score': result.overall_score,
                'passed_risk_checks': result.passed_risk_checks,
                'best_market_condition': self._determine_best_market_condition(
                    sharpe, total_return, max_dd, market_conditions
                )
            }
        
        # 按市场条件分类推荐
        recommendations = self._generate_recommendations(strategy_analysis, market_conditions)
        
        return {
            'market_conditions': market_conditions,
            'strategy_performance': strategy_analysis,
            'recommendations': recommendations,
            'best_overall': max(strategy_analysis.items(), 
                               key=lambda x: x[1]['overall_score'])[0] if strategy_analysis else None
        }
    
    def _identify_market_conditions(self, data: pd.DataFrame) -> MarketCondition:
        """识别市场条件"""
        if 'close' not in data.columns or len(data) < 20:
            return MarketCondition('medium', 'sideways', 'medium', 'mean_reverting')
        
        # 计算波动率
        returns = data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        if volatility < 0.15:
            vol_level = 'low'
        elif volatility > 0.30:
            vol_level = 'high'
        else:
            vol_level = 'medium'
        
        # 计算趋势
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        if price_change > 0.1:
            trend = 'bull'
        elif price_change < -0.1:
            trend = 'bear'
        else:
            trend = 'sideways'
        
        # 计算成交量（如果有）
        if 'quantity' in data.columns:
            avg_volume = data['quantity'].mean()
            recent_volume = data['quantity'].tail(10).mean()
            if recent_volume > avg_volume * 1.2:
                volume_level = 'high'
            elif recent_volume < avg_volume * 0.8:
                volume_level = 'low'
            else:
                volume_level = 'medium'
        else:
            volume_level = 'medium'
        
        # 确定市场状态
        if volatility > 0.25:
            regime = 'volatile'
        elif abs(price_change) > 0.15:
            regime = 'trending'
        else:
            regime = 'mean_reverting'
        
        return MarketCondition(vol_level, trend, volume_level, regime)
    
    def _get_strategy_category(self, strategy_name: str) -> str:
        """获取策略分类"""
        if strategy_name.startswith('ml_'):
            return 'ML'
        elif strategy_name.startswith('rl_'):
            return 'RL'
        elif strategy_name.startswith('hft_'):
            return 'HFT'
        elif strategy_name.startswith('statistical_'):
            return 'Statistical'
        else:
            return 'Classical'
    
    def _determine_best_market_condition(
        self,
        sharpe: float,
        total_return: float,
        max_drawdown: float,
        market_condition: MarketCondition
    ) -> str:
        """确定策略最适合的市场条件"""
        # 基于策略特征判断
        if sharpe > 2.0 and total_return > 20:
            if market_condition.regime == 'trending':
                return 'Trending markets'
            elif market_condition.regime == 'mean_reverting':
                return 'Mean-reverting markets'
        elif sharpe > 1.0 and max_drawdown < 10:
            return 'Stable markets'
        else:
            return 'Various market conditions'
    
    def _generate_recommendations(
        self,
        strategy_analysis: Dict,
        market_condition: MarketCondition
    ) -> Dict[str, List[str]]:
        """生成推荐"""
        recommendations = {
            'best_overall': [],
            'best_sharpe': [],
            'best_return': [],
            'lowest_risk': [],
            'best_for_current_market': []
        }
        
        if not strategy_analysis:
            return recommendations
        
        # 最佳综合评分
        best_overall = max(strategy_analysis.items(), key=lambda x: x[1]['overall_score'])
        recommendations['best_overall'] = [best_overall[0]]
        
        # 最佳Sharpe
        best_sharpe = max(strategy_analysis.items(), key=lambda x: x[1]['sharpe_ratio'])
        recommendations['best_sharpe'] = [best_sharpe[0]]
        
        # 最佳收益
        best_return = max(strategy_analysis.items(), key=lambda x: x[1]['total_return'])
        recommendations['best_return'] = [best_return[0]]
        
        # 最低风险（最小回撤）
        lowest_risk = min(strategy_analysis.items(), key=lambda x: abs(x[1]['max_drawdown']))
        recommendations['lowest_risk'] = [lowest_risk[0]]
        
        # 根据当前市场条件推荐
        if market_condition.regime == 'trending':
            # 趋势市场：推荐动量策略
            trending_strategies = [
                s for s, info in strategy_analysis.items()
                if 'momentum' in s.lower() or 'ma_crossover' in s.lower()
            ]
            if trending_strategies:
                recommendations['best_for_current_market'] = trending_strategies[:3]
        elif market_condition.regime == 'mean_reverting':
            # 均值回归市场：推荐均值回归策略
            mean_reversion_strategies = [
                s for s, info in strategy_analysis.items()
                if 'mean_reversion' in s.lower() or 'pairs' in s.lower()
            ]
            if mean_reversion_strategies:
                recommendations['best_for_current_market'] = mean_reversion_strategies[:3]
        elif market_condition.regime == 'volatile':
            # 波动市场：推荐低风险策略
            low_risk_strategies = [
                s for s, info in strategy_analysis.items()
                if abs(info['max_drawdown']) < 15
            ]
            if low_risk_strategies:
                recommendations['best_for_current_market'] = low_risk_strategies[:3]
        
        return recommendations
    
    def print_analysis_report(self, analysis_result: Dict):
        """打印分析报告"""
        print("\n" + "="*80)
        print("Strategy Performance Analysis")
        print("="*80)
        
        # 市场条件
        mc = analysis_result['market_conditions']
        print(f"\nMarket Conditions:")
        print(f"  Volatility: {mc.volatility.upper()}")
        print(f"  Trend: {mc.trend.upper()}")
        print(f"  Volume: {mc.volume.upper()}")
        print(f"  Regime: {mc.regime.upper()}")
        
        # 策略表现
        print(f"\nStrategy Performance by Category:")
        categories = {}
        for name, perf in analysis_result['strategy_performance'].items():
            cat = perf['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, perf))
        
        for cat, strategies in categories.items():
            print(f"\n  {cat}:")
            for name, perf in sorted(strategies, key=lambda x: x[1]['overall_score'], reverse=True)[:3]:
                print(f"    {name}:")
                print(f"      Score: {perf['overall_score']:.2f}/100")
                print(f"      Sharpe: {perf['sharpe_ratio']:.3f}")
                print(f"      Return: {perf['total_return']:.2f}%")
                print(f"      Max DD: {perf['max_drawdown']:.2f}%")
        
        # 推荐
        rec = analysis_result['recommendations']
        print(f"\nRecommendations:")
        print(f"  Best Overall: {', '.join(rec['best_overall'])}")
        print(f"  Best Sharpe: {', '.join(rec['best_sharpe'])}")
        print(f"  Best Return: {', '.join(rec['best_return'])}")
        print(f"  Lowest Risk: {', '.join(rec['lowest_risk'])}")
        if rec['best_for_current_market']:
            print(f"  Best for Current Market: {', '.join(rec['best_for_current_market'])}")
        
        print("="*80)








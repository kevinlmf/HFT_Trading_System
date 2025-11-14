"""
完整交易流程演示
展示：EDA -> 数据清理 -> 智能执行 -> 策略对比 -> 风控 -> 评估
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from Execution.engine.complete_trading_flow import CompleteTradingFlow, create_sample_strategies
from Execution.engine.pipeline import create_sample_data


def create_market_data(n_days: int = 252) -> pd.DataFrame:
    """创建市场数据用于策略测试"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # 生成价格数据
    prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(n_days) * 0.5,
        'high': prices + np.abs(np.random.randn(n_days) * 1),
        'low': prices - np.abs(np.random.randn(n_days) * 1),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    data.set_index('date', inplace=True)
    return data


def main():
    """主演示函数"""
    print("="*80)
    print("Complete Trading Flow Demo")
    print("="*80)
    print("\n本演示将展示完整的交易流程：")
    print("1. EDA分析 + 数据清理 + 智能执行")
    print("2. 策略对比（Monte Carlo + Backtest）")
    print("3. 风控（VaR/CVaR）")
    print("4. 性能评估（Sharpe Ratio）")
    print("5. 综合评分和推荐")
    print("\n" + "="*80)
    
    # 创建市场数据
    print("\n生成市场数据...")
    market_data = create_market_data(n_days=252)
    
    # 创建策略
    print("创建策略...")
    strategies = create_sample_strategies()
    
    # 创建完整流程
    flow = CompleteTradingFlow(
        risk_free_rate=0.02,
        periods_per_year=252,
        monte_carlo_paths=50000  # 使用较少路径以加快演示
    )
    
    # 运行完整流程
    print("\n开始执行完整流程...")
    result = flow.execute_complete_flow(
        data=market_data,
        strategies=strategies,
        risk_limits={
            'max_drawdown': 0.20,  # 20%
            'max_var_95': 0.05,    # 5%
            'max_cvar_95': 0.08,   # 8%
            'min_sharpe': 0.5
        }
    )
    
    # 打印最终推荐
    print("\n" + "="*80)
    print("Final Recommendation")
    print("="*80)
    if result['recommended_strategy']:
        recommended = result['recommended_strategy']
        print(f"\n推荐策略: {recommended.strategy_name}")
        print(f"综合评分: {recommended.overall_score:.2f}/100")
        print(f"Sharpe Ratio: {recommended.sharpe_ratio:.3f}")
        print(f"风险检查: {'通过' if recommended.passed_risk_checks else '未通过'}")
        
        if recommended.risk_metrics:
            print(f"\n风险指标:")
            print(f"  最大回撤: {recommended.risk_metrics.max_drawdown*100:.2f}%")
            print(f"  VaR (95%): {recommended.risk_metrics.var_95*100:.2f}%")
            print(f"  CVaR (95%): {recommended.risk_metrics.cvar_95*100:.2f}%")
    
    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)


if __name__ == "__main__":
    main()








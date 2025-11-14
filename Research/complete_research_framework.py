"""
Complete Research Framework Integration

整合完整的量化研究流程：
1. Market Microstructure Profiling
2. Factor Hypothesis Generation
3. Statistical Validation
4. Machine Learning Validation

核心思想：先经济学解释 → 再统计学验证 → 最后算法集成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Research.microstructure_profiling import MicrostructureProfiler
from Research.factor_hypothesis import FactorHypothesisGenerator
from Research.statistical_validation import StatisticalValidator
from Research.ml_validation import MLValidator


class CompleteResearchFramework:
    """
    完整的量化研究框架
    
    实现：Market Microstructure Profiling → Hypothesis → Validation
    """
    
    def __init__(self):
        self.profiler = MicrostructureProfiler()
        self.hypothesis_generator = FactorHypothesisGenerator(self.profiler)
        self.statistical_validator = StatisticalValidator()
        self.ml_validator = MLValidator()
        
        self.profiling_results = {}
        self.hypotheses = []
        self.validation_results = {}
    
    def run_complete_research_pipeline(
        self,
        market_data: Dict[str, pd.DataFrame],
        forward_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        运行完整的研究流程
        
        Args:
            market_data: 市场数据字典
            forward_returns: 未来收益序列（用于验证）
        
        Returns:
            完整研究结果
        """
        results = {}
        
        # Step 1: Market Microstructure Profiling
        print("=" * 80)
        print("Step 1: Market Microstructure Profiling")
        print("=" * 80)
        
        profiling_results = self.profiler.comprehensive_profile(market_data)
        self.profiling_results = profiling_results
        
        # 生成经济洞察
        insights = self.profiler.generate_economic_insights()
        results['profiling'] = {
            'metrics': profiling_results,
            'insights': insights
        }
        
        print("\nProfiling Results:")
        for key, value in profiling_results.items():
            print(f"  {key}: ✓")
        
        print("\nEconomic Insights:")
        for key, insight in insights.items():
            print(f"  • {insight}")
        
        # Step 2: Factor Hypothesis Generation
        print("\n" + "=" * 80)
        print("Step 2: Factor Hypothesis Generation")
        print("=" * 80)
        
        hypotheses = self.hypothesis_generator.generate_all_hypotheses_from_profiling(
            profiling_results
        )
        self.hypotheses = hypotheses
        
        self.hypothesis_generator.print_hypotheses()
        
        # Step 3: Statistical Validation
        if forward_returns is not None:
            print("\n" + "=" * 80)
            print("Step 3: Statistical Validation")
            print("=" * 80)
            
            statistical_results = {}
            for hyp in hypotheses:
                # 计算因子值
                factor_values = self.hypothesis_generator.compute_factor_values(
                    market_data, hyp
                )
                
                if len(factor_values) > 0:
                    # 统计验证
                    stat_result = self.statistical_validator.comprehensive_validation(
                        factor_values, forward_returns, hyp
                    )
                    statistical_results[hyp.name] = stat_result
                    
                    self.statistical_validator.print_validation_results(hyp.name)
            
            results['statistical_validation'] = statistical_results
        
        # Step 4: Machine Learning Validation
        if forward_returns is not None:
            print("\n" + "=" * 80)
            print("Step 4: Machine Learning Validation")
            print("=" * 80)
            
            ml_results = {}
            baseline_features = None  # 可以添加基准特征
            
            for hyp in hypotheses:
                # 计算因子值
                factor_values = self.hypothesis_generator.compute_factor_values(
                    market_data, hyp
                )
                
                if len(factor_values) > 0:
                    # ML验证
                    ml_result = self.ml_validator.comprehensive_ml_validation(
                        factor_values, forward_returns, hyp, baseline_features
                    )
                    ml_results[hyp.name] = ml_result
                    
                    self.ml_validator.print_ml_validation_results(hyp.name)
            
            results['ml_validation'] = ml_results
        
        # Step 5: Final Summary
        print("\n" + "=" * 80)
        print("Research Summary")
        print("=" * 80)
        
        valid_factors = []
        for hyp in hypotheses:
            is_stat_valid = False
            is_ml_valid = False
            
            if 'statistical_validation' in results and hyp.name in results['statistical_validation']:
                is_stat_valid = results['statistical_validation'][hyp.name]['is_valid']
            
            if 'ml_validation' in results and hyp.name in results['ml_validation']:
                is_ml_valid = results['ml_validation'][hyp.name]['is_valid']
            
            if is_stat_valid or is_ml_valid:
                valid_factors.append({
                    'name': hyp.name,
                    'category': hyp.category.value,
                    'statistical_valid': is_stat_valid,
                    'ml_valid': is_ml_valid,
                    'economic_motivation': hyp.economic_motivation
                })
        
        print(f"\nValid Factors: {len(valid_factors)}")
        for factor in valid_factors:
            print(f"  ✓ {factor['name']} ({factor['category']})")
            print(f"    Statistical: {'✓' if factor['statistical_valid'] else '✗'}")
            print(f"    ML: {'✓' if factor['ml_valid'] else '✗'}")
        
        results['summary'] = {
            'total_hypotheses': len(hypotheses),
            'valid_factors': valid_factors,
            'research_complete': True
        }
        
        self.validation_results = results
        return results
    
    def export_results(self, output_path: str):
        """导出研究结果"""
        import json
        
        export_data = {
            'profiling': {
                k: {attr: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                    for attr, v in val.__dict__.items() if not isinstance(v, np.ndarray)}
                for k, val in self.profiling_results.items()
            },
            'hypotheses': [
                {
                    'name': h.name,
                    'category': h.category.value,
                    'economic_motivation': h.economic_motivation,
                    'formula': h.formula,
                    'expected_direction': h.expected_direction
                }
                for h in self.hypotheses
            ],
            'summary': self.validation_results.get('summary', {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults exported to: {output_path}")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1min')
    
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    returns = pd.Series(prices).pct_change()
    forward_returns = returns.shift(-1).dropna()  # 未来收益
    
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
    
    market_data = {
        'prices': pd.Series(prices, index=dates),
        'bid_prices': pd.Series(bid_prices, index=dates),
        'ask_prices': pd.Series(ask_prices, index=dates),
        'bid_sizes': pd.Series(bid_sizes, index=dates),
        'ask_sizes': pd.Series(ask_sizes, index=dates),
        'trades': trades,
        'returns': returns
    }
    
    # 运行完整研究流程
    framework = CompleteResearchFramework()
    results = framework.run_complete_research_pipeline(market_data, forward_returns)
    
    # 导出结果
    framework.export_results('research_results.json')


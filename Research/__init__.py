"""
Research Module - 量化研究框架

实现正确的量化研究范式：
Market Microstructure Profiling → Factor Hypothesis → Validation

核心思想：先经济学解释 → 再统计学验证 → 最后算法集成
"""

from .microstructure_profiling import (
    MicrostructureProfiler,
    PriceFormationMetrics,
    LiquidityMetrics,
    OrderFlowMetrics,
    MarketImpactMetrics,
    LatencyMetrics
)

from .factor_hypothesis import (
    FactorHypothesisGenerator,
    FactorHypothesis,
    FactorCategory
)

from .statistical_validation import (
    StatisticalValidator,
    RegressionResult,
    LongShortResult,
    StabilityResult
)

from .ml_validation import (
    MLValidator,
    MLValidationResult,
    RLValidationResult
)

from .complete_research_framework import CompleteResearchFramework

__all__ = [
    # Profiling
    'MicrostructureProfiler',
    'PriceFormationMetrics',
    'LiquidityMetrics',
    'OrderFlowMetrics',
    'MarketImpactMetrics',
    'LatencyMetrics',
    
    # Hypothesis
    'FactorHypothesisGenerator',
    'FactorHypothesis',
    'FactorCategory',
    
    # Validation
    'StatisticalValidator',
    'RegressionResult',
    'LongShortResult',
    'StabilityResult',
    'MLValidator',
    'MLValidationResult',
    'RLValidationResult',
    
    # Complete Framework
    'CompleteResearchFramework'
]














"""
04. Alpha Modeling - 信号构建模块

把 microstructure → 转换成 可预测未来方向或风险的 alpha。

产出：alpha_t（预测信号）
"""

try:
    from .factor_hypothesis import FactorHypothesisGenerator, FactorHypothesis
    from .statistical_validation import StatisticalValidator
    from .ml_validation import MLValidator
    __all__ = [
        'FactorHypothesisGenerator',
        'FactorHypothesis',
        'StatisticalValidator',
        'MLValidator',
    ]
except ImportError:
    __all__ = []


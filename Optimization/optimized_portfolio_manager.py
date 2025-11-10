"""
使用深度优化栈的优化投资组合管理器
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Optimization.optimization_stack import (
    OptimizationStack,
    ModelObjective,
    ExecutionBackend
)
from Execution.risk_control.portfolio_manager import (
    AdvancedPortfolioManager,
    RiskModel,
    RiskConstraints
)


class OptimizedPortfolioManager(AdvancedPortfolioManager):
    """
    使用深度优化栈的优化投资组合管理器
    
    继承自AdvancedPortfolioManager，但使用优化栈进行更高效的优化
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        risk_model: RiskModel = RiskModel.RISK_PARITY,
        constraints: Optional[RiskConstraints] = None,
        use_optimization_stack: bool = True
    ):
        super().__init__(initial_capital, risk_model, constraints)
        self.use_optimization_stack = use_optimization_stack
        if use_optimization_stack:
            self.optimization_stack = OptimizationStack()
        else:
            self.optimization_stack = None
    
    def calculate_optimal_weights(
        self,
        price_data: Dict[str, pd.DataFrame],
        use_optimization_stack: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        计算最优权重（使用优化栈）
        
        Args:
            price_data: 价格数据字典
            use_optimization_stack: 是否使用优化栈（覆盖初始化设置）
        
        Returns:
            最优权重字典
        """
        if use_optimization_stack is None:
            use_optimization_stack = self.use_optimization_stack
        
        if use_optimization_stack and self.optimization_stack:
            return self._calculate_with_optimization_stack(price_data)
        else:
            # 使用父类的原始方法
            return super().calculate_optimal_weights(price_data)
    
    def _calculate_with_optimization_stack(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """使用优化栈计算最优权重"""
        # 准备收益率数据
        returns_list = []
        symbols = []
        
        for symbol, df in price_data.items():
            if 'close' in df.columns and len(df) > 1:
                prices = df['close']
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    returns_list.append(returns.values)
                    symbols.append(symbol)
        
        if len(returns_list) == 0:
            return {}
        
        # 对齐时间序列
        min_length = min(len(r) for r in returns_list)
        returns_array = np.array([r[-min_length:] for r in returns_list]).T
        
        # 根据风险模型选择优化目标
        if self.risk_model == RiskModel.MEAN_VARIANCE:
            objective = ModelObjective.MAXIMIZE_SHARPE
        elif self.risk_model == RiskModel.RISK_PARITY:
            objective = ModelObjective.MINIMIZE_VARIANCE
        elif self.risk_model == RiskModel.INVERSE_VOLATILITY:
            objective = ModelObjective.MINIMIZE_VARIANCE
        else:
            objective = ModelObjective.MAXIMIZE_SHARPE
        
        # 定义约束
        constraints = {}
        if self.constraints:
            constraints['max_position_size'] = self.constraints.max_position_size
            constraints['min_position_size'] = 0.0
            constraints['sum_to_one'] = True
        
        # 使用优化栈优化
        result = self.optimization_stack.optimize_portfolio(
            returns_array,
            objective=objective,
            constraints=constraints
        )
        
        # 转换为字典
        optimal_weights = {}
        for i, symbol in enumerate(symbols):
            optimal_weights[symbol] = float(result['optimal_weights'][i])
        
        # 应用约束
        optimal_weights = self._apply_constraints(optimal_weights)
        
        return optimal_weights
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用约束条件"""
        if not self.constraints:
            return weights
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        # 最大仓位限制
        if self.constraints.max_position_size < 1.0:
            for symbol in weights:
                weights[symbol] = min(weights[symbol], self.constraints.max_position_size)
            # 重新归一化
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def get_optimization_info(self) -> Dict[str, any]:
        """获取优化信息"""
        if self.optimization_stack:
            return {
                'using_optimization_stack': True,
                'gpu_available': self.optimization_stack.system_layer.gpu_available,
                'backend': self.optimization_stack.system_layer.backend.value
            }
        return {'using_optimization_stack': False}







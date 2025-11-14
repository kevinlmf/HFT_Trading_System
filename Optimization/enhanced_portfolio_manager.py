"""
优化的OptimizedPortfolioManager - 集成QDB和优化方法/数据结构
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Optimization.optimized_optimization_stack import (
    EnhancedOptimizationStack,
    OptimizedDataLoader,
    ModelObjective
)
from Execution.risk_control.portfolio_manager import (
    AdvancedPortfolioManager,
    RiskModel,
    RiskConstraints
)


class EnhancedOptimizedPortfolioManager(AdvancedPortfolioManager):
    """
    增强的优化投资组合管理器
    
    优化：
    1. 集成QDB快速数据加载
    2. 优化的协方差矩阵计算（缓存）
    3. 智能算法选择
    4. 优化的数据结构（float32，内存对齐）
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        risk_model: RiskModel = RiskModel.RISK_PARITY,
        constraints: Optional[RiskConstraints] = None,
        use_qdb: bool = True,
        qdb: Optional[object] = None
    ):
        super().__init__(initial_capital, risk_model, constraints)
        
        # 使用增强的优化栈（集成QDB）
        self.optimization_stack = EnhancedOptimizationStack(
            use_qdb=use_qdb,
            qdb=qdb
        )
    
    def calculate_optimal_weights(
        self,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        symbols: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        use_qdb: bool = True
    ) -> Dict[str, float]:
        """
        计算最优权重（优化版本）
        
        支持两种模式：
        1. QDB模式：直接从QDB加载数据（推荐）
        2. 传统模式：从price_data字典加载
        
        Args:
            price_data: 价格数据字典（传统模式）
            symbols: 交易标的列表（QDB模式）
            start_time: 开始时间（QDB模式）
            end_time: 结束时间（QDB模式）
            use_qdb: 是否使用QDB
        
        Returns:
            最优权重字典
        """
        if use_qdb and symbols and self.optimization_stack.data_loader.use_qdb:
            # QDB模式：快速加载和优化
            return self._calculate_with_qdb(symbols, start_time, end_time)
        elif price_data:
            # 传统模式：从price_data加载
            return self._calculate_with_price_data(price_data)
        else:
            raise ValueError("Either provide symbols (for QDB) or price_data (for traditional)")
    
    def _calculate_with_qdb(
        self,
        symbols: List[str],
        start_time: Optional[str],
        end_time: Optional[str]
    ) -> Dict[str, float]:
        """使用QDB计算最优权重"""
        # 根据风险模型选择优化目标
        objective_map = {
            RiskModel.MEAN_VARIANCE: ModelObjective.MAXIMIZE_SHARPE,
            RiskModel.RISK_PARITY: ModelObjective.MINIMIZE_VARIANCE,
            RiskModel.INVERSE_VOLATILITY: ModelObjective.MINIMIZE_VARIANCE,
        }
        objective = objective_map.get(self.risk_model, ModelObjective.MAXIMIZE_SHARPE)
        
        # 定义约束
        constraints = {}
        if self.constraints:
            constraints['max_position_size'] = self.constraints.max_position_size
            constraints['min_position_size'] = 0.0
            constraints['sum_to_one'] = True
        
        # 使用增强的优化栈优化
        result = self.optimization_stack.optimize_portfolio_from_qdb(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            objective=objective,
            constraints=constraints
        )
        
        if 'error' in result:
            return {}
        
        # 应用约束
        optimal_weights = result['optimal_weights']
        optimal_weights = self._apply_constraints(optimal_weights)
        
        return optimal_weights
    
    def _calculate_with_price_data(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """使用传统price_data计算最优权重"""
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
        
        # 使用优化栈优化（传统方式）
        objective_map = {
            RiskModel.MEAN_VARIANCE: ModelObjective.MAXIMIZE_SHARPE,
            RiskModel.RISK_PARITY: ModelObjective.MINIMIZE_VARIANCE,
            RiskModel.INVERSE_VOLATILITY: ModelObjective.MINIMIZE_VARIANCE,
        }
        objective = objective_map.get(self.risk_model, ModelObjective.MAXIMIZE_SHARPE)
        
        constraints = {}
        if self.constraints:
            constraints['max_position_size'] = self.constraints.max_position_size
            constraints['sum_to_one'] = True
        
        result = self.optimization_stack.optimize_portfolio(
            returns_array,
            objective=objective,
            constraints=constraints
        )
        
        optimal_weights = {}
        for i, symbol in enumerate(symbols):
            optimal_weights[symbol] = float(result['optimal_weights'][i])
        
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
        info = {
            'using_enhanced_stack': True,
            'qdb_enabled': self.optimization_stack.data_loader.use_qdb,
            'gpu_available': self.optimization_stack.system_layer.gpu_available,
            'cache_size': len(self.optimization_stack.data_loader._covariance_cache)
        }
        return info
    
    def clear_cache(self):
        """清空缓存"""
        self.optimization_stack.data_loader.clear_cache()














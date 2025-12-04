"""
CVXPY Optimizer - Convex Optimization for Portfolio Management
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not installed. Please install using `pip install cvxpy`.")

from Optimization.optimization_types import ModelObjective, AlgorithmType, AlgorithmConfig


class CVXPYOptimizer:
    """
    基于CVXPY的凸优化器
    
    优势：
    1. 速度快：使用专用求解器（OSQP, SCS, ECOS）
    2. 稳定性：凸优化保证全局最优
    3. 灵活性：易于添加各种线性/二次约束
    """
    
    def __init__(self, solver: str = 'OSQP'):
        """
        Args:
            solver: 求解器名称 ('OSQP', 'SCS', 'ECOS', 'CLARABEL')
        """
        self.solver = solver
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for this optimizer")
            
    def optimize_portfolio(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        objective: ModelObjective = ModelObjective.MAXIMIZE_SHARPE,
        constraints: Optional[Dict] = None,
        risk_aversion: float = 1.0
    ) -> Dict[str, Any]:
        """
        执行投资组合优化
        
        Args:
            mean_returns: 预期收益率向量 (N,)
            cov_matrix: 协方差矩阵 (N, N)
            objective: 优化目标
            constraints: 约束条件
            risk_aversion: 风险厌恶系数 (用于均值-方差模型)
            
        Returns:
            优化结果字典
        """
        n_assets = len(mean_returns)
        
        # 定义优化变量
        w = cp.Variable(n_assets)
        
        # 定义约束
        cons = [cp.sum(w) == 1.0, w >= 0.0]  # 默认：全额投资，无做空
        
        if constraints:
            if 'max_position' in constraints:
                cons.append(w <= constraints['max_position'])
            if 'min_position' in constraints:
                cons.append(w >= constraints['min_position'])
            # 可以添加更多约束...
            
        # 定义目标函数
        if objective == ModelObjective.MINIMIZE_VARIANCE:
            # 最小化方差: min w^T * Sigma * w
            # CVXPY: cp.quad_form(w, cov_matrix)
            obj = cp.Minimize(cp.quad_form(w, cov_matrix))
            
        elif objective == ModelObjective.MAXIMIZE_RETURN:
            # 最大化收益 (通常需要风险约束，否则会全仓压在收益最高的资产)
            obj = cp.Maximize(mean_returns @ w)
            
        elif objective == ModelObjective.MAXIMIZE_SHARPE:
            # 最大化夏普比率 (Max Sharpe)
            # 这是一个非凸问题，但可以转化为凸问题 (SOCP 或 QP)
            # 变换：y = w / (w^T * mu), 最小化 y^T * Sigma * y
            # 这里我们使用简单的 均值-方差 权衡作为近似，或者使用二次规划求解
            
            # 方法1：最大化 mu^T w - lambda * w^T Sigma w (Quadratic Utility)
            # 这不是直接最大化Sharpe，但对于特定的lambda，它对应于有效前沿上的一点
            obj = cp.Maximize(mean_returns @ w - risk_aversion * cp.quad_form(w, cov_matrix))
            
        else:
            raise ValueError(f"Unsupported objective: {objective}")
            
        # 构建问题
        prob = cp.Problem(obj, cons)
        
        # 求解
        try:
            prob.solve(solver=self.solver, verbose=False)
        except Exception as e:
            # 如果默认求解器失败，尝试SCS
            if self.solver != 'SCS':
                prob.solve(solver='SCS', verbose=False)
            else:
                raise e
                
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {
                'success': False,
                'status': prob.status,
                'weights': np.zeros(n_assets)
            }
            
        # 提取结果
        optimal_weights = w.value
        
        # 清理小的数值误差
        optimal_weights[np.abs(optimal_weights) < 1e-6] = 0.0
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # 计算指标
        port_return = np.dot(optimal_weights, mean_returns)
        port_var = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        port_std = np.sqrt(port_var)
        sharpe = port_return / port_std if port_std > 0 else 0
        
        return {
            'success': True,
            'status': prob.status,
            'weights': optimal_weights,
            'performance': {
                'return': port_return,
                'volatility': port_std,
                'sharpe': sharpe
            },
            'solver_stats': {
                'solver': prob.solver_stats.solver_name if prob.solver_stats else self.solver,
                'setup_time': prob.solver_stats.setup_time if prob.solver_stats else 0,
                'solve_time': prob.solver_stats.solve_time if prob.solver_stats else 0,
                'iterations': prob.solver_stats.num_iters if prob.solver_stats else 0
            }
        }

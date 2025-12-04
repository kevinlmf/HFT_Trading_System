"""
JAX Optimizer - High Performance Optimization using JAX and Factor Models
"""
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    from jax.scipy.optimize import minimize as jax_minimize
    import jax.scipy.linalg as jlinalg
    
    # Enable 64-bit precision for better numerical stability in finance
    jax.config.update("jax_enable_x64", True)
    
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not installed. Please install using `pip install jax jaxlib`.")

from Optimization.optimization_types import ModelObjective, AlgorithmType

class JAXOptimizer:
    """
    基于JAX的高性能优化器
    
    特性：
    1. GPU加速：自动利用GPU进行矩阵运算
    2. JIT编译：即时编译核心函数，大幅提升速度
    3. 因子模型：支持结构化协方差矩阵 (B*F*B.T + D)，降低复杂度
    4. Cholesky分解：数值稳定的线性方程组求解
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for this optimizer")
            
    @staticmethod
    @jit
    def _factor_model_inverse(B: jnp.ndarray, F: jnp.ndarray, D_inv_diag: jnp.ndarray) -> jnp.ndarray:
        """
        使用Woodbury矩阵恒等式计算因子模型协方差矩阵的逆
        Sigma = B F B^T + D
        Sigma^-1 = D^-1 - D^-1 B (F^-1 + B^T D^-1 B)^-1 B^T D^-1
        
        Args:
            B: 因子载荷 (N, K)
            F: 因子协方差 (K, K)
            D_inv_diag: 特异风险逆的对角线 (N,)
            
        Returns:
            Sigma^-1 (N, N) - 注意：对于超大规模问题，通常不直接返回这个矩阵，而是提供solve函数
        """
        N, K = B.shape
        
        # 1. 计算中间项 M = (F^-1 + B^T D^-1 B)
        # 为了数值稳定性，我们不直接求F^-1，而是解方程
        # 但这里假设K很小，直接求逆也可以，或者使用Cholesky
        
        # B_tilde = D^-1/2 B
        D_inv_sqrt = jnp.sqrt(D_inv_diag)
        B_tilde = B * D_inv_sqrt[:, None]  # Broadcasting
        
        # M = F^-1 + B_tilde^T B_tilde
        # 如果F是单位阵(PCA)，则F^-1 = I
        # 这里通用处理：
        F_inv = jnp.linalg.inv(F)
        M = F_inv + B_tilde.T @ B_tilde
        
        # 2. 计算 M^-1
        # 使用Cholesky分解求逆: M = L L^T
        L = jlinalg.cho_factor(M, lower=True)
        M_inv = jlinalg.cho_solve(L, jnp.eye(K))
        
        # 3. 组合结果 (这一步是O(N^2 K)，如果只需要solve，可以不显式计算)
        # Sigma^-1 = diag(D^-1) - (D^-1 B) M^-1 (B^T D^-1)
        
        # 这里我们返回用于solve的组件，而不是完整的逆矩阵，以节省内存
        return M_inv
        
    @staticmethod
    @jit
    def _solve_linear_system_factor_model(
        B: jnp.ndarray, 
        F: jnp.ndarray, 
        D_diag: jnp.ndarray, 
        b: jnp.ndarray
    ) -> jnp.ndarray:
        """
        求解 Sigma * x = b，利用因子结构
        使用Woodbury公式: x = Sigma^-1 b
        x = D^-1 b - D^-1 B (F^-1 + B^T D^-1 B)^-1 B^T D^-1 b
        
        复杂度: O(NK + K^3) vs O(N^3)
        """
        D_inv_diag = 1.0 / D_diag
        
        # 1. v = D^-1 b
        v = b * D_inv_diag
        
        # 2. 计算 M = F^-1 + B^T D^-1 B
        D_inv_sqrt = jnp.sqrt(D_inv_diag)
        B_tilde = B * D_inv_sqrt[:, None]
        
        F_inv = jnp.linalg.inv(F) # KxK, K is small
        M = F_inv + B_tilde.T @ B_tilde
        
        # 3. 解方程 M z = B^T v
        rhs = B.T @ v
        L = jlinalg.cho_factor(M, lower=True)
        z = jlinalg.cho_solve(L, rhs)
        
        # 4. x = v - D^-1 B z
        x = v - (B @ z) * D_inv_diag
        
        return x

    @partial(jit, static_argnums=(0,))
    def optimize_markowitz_analytic(
        self,
        mean_returns: jnp.ndarray,
        cov_matrix: Optional[jnp.ndarray] = None,
        B: Optional[jnp.ndarray] = None,
        F: Optional[jnp.ndarray] = None,
        D: Optional[jnp.ndarray] = None,
        risk_aversion: float = 1.0
    ) -> Dict[str, Any]:
        """
        解析解求解无约束Markowitz问题 (Maximize Utility)
        max w^T mu - lambda/2 w^T Sigma w
        Solution: w = (1/lambda) Sigma^-1 mu
        (注意：这里没有 sum(w)=1 约束，通常用于多空组合)
        
        如果有 sum(w)=1 约束:
        w = Sigma^-1 (mu + nu * 1) / lambda
        其中 nu 是拉格朗日乘子
        """
        N = mean_returns.shape[0]
        ones = jnp.ones(N)
        
        if B is not None and F is not None and D is not None:
            # 使用因子模型求解
            # 1. Solve Sigma * w_unc = mu
            w_unc = self._solve_linear_system_factor_model(B, F, D, mean_returns)
            # 2. Solve Sigma * w_ones = 1
            w_ones = self._solve_linear_system_factor_model(B, F, D, ones)
        else:
            # 使用Cholesky分解求解一般矩阵
            L = jlinalg.cho_factor(cov_matrix, lower=True)
            w_unc = jlinalg.cho_solve(L, mean_returns)
            w_ones = jlinalg.cho_solve(L, ones)
            
        # 计算带约束的解 (sum(w) = 1)
        # w = (Sigma^-1 1) / (1^T Sigma^-1 1)  (最小方差组合)
        # 或者切线组合 (最大夏普)
        # w = (Sigma^-1 mu) / (1^T Sigma^-1 mu)
        
        # 这里我们实现最大夏普比率组合 (Tangency Portfolio)
        # 假设无风险利率为0
        denominator = jnp.dot(ones, w_unc)
        weights = w_unc / denominator
        
        return weights

    def optimize_portfolio(
        self,
        mean_returns: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        factor_loadings: Optional[np.ndarray] = None, # B
        factor_cov: Optional[np.ndarray] = None,      # F
        idiosyncratic_risk: Optional[np.ndarray] = None, # D (diagonal)
        objective: ModelObjective = ModelObjective.MAXIMIZE_SHARPE,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        JAX优化入口
        """
        # 转换为JAX数组
        mu = jnp.array(mean_returns)
        
        use_factor_model = (factor_loadings is not None and 
                          factor_cov is not None and 
                          idiosyncratic_risk is not None)
        
        if use_factor_model:
            B = jnp.array(factor_loadings)
            F = jnp.array(factor_cov)
            D = jnp.array(idiosyncratic_risk)
            Sigma = None
        else:
            B, F, D = None, None, None
            Sigma = jnp.array(cov_matrix)
            
        # 目前仅支持解析解路径 (无不等式约束)
        # 如果有复杂约束，需要使用 jax.scipy.optimize.minimize 或 projected gradient descent
        
        has_inequality_constraints = constraints and ('max_position' in constraints or 'min_position' in constraints)
        
        if not has_inequality_constraints and objective == ModelObjective.MAXIMIZE_SHARPE:
            # 使用解析解 (极快)
            weights = self.optimize_markowitz_analytic(mu, Sigma, B, F, D)
            success = True
            message = "Analytic solution (Tangency Portfolio)"
        else:
            # 使用数值优化 (Projected Gradient Descent)
            # TODO: 实现支持约束的PGD
            warnings.warn("Inequality constraints not yet supported in JAX optimizer, ignoring.")
            weights = self.optimize_markowitz_analytic(mu, Sigma, B, F, D)
            success = True
            message = "Analytic solution (Constraints ignored)"
            
        # 转换回NumPy
        weights_np = np.array(weights)
        
        # 计算性能指标
        port_return = float(np.dot(weights_np, mean_returns))
        
        if use_factor_model:
            # Var = w^T (B F B^T + D) w = (w^T B) F (B^T w) + w^T D w
            w_B = np.dot(weights_np, factor_loadings)
            factor_var = np.dot(w_B, np.dot(factor_cov, w_B))
            idio_var = np.dot(weights_np**2, idiosyncratic_risk)
            port_var = factor_var + idio_var
        else:
            port_var = float(np.dot(weights_np, np.dot(cov_matrix, weights_np)))
            
        port_std = np.sqrt(port_var)
        sharpe = port_return / port_std if port_std > 0 else 0
        
        return {
            'success': success,
            'weights': weights_np,
            'optimization_info': {
                'algorithm': 'JAX_FactorModel' if use_factor_model else 'JAX_Cholesky',
                'message': message,
                'device': str(jax.devices()[0])
            },
            'performance_metrics': {
                'portfolio_return': port_return,
                'portfolio_std': port_std,
                'sharpe_ratio': sharpe
            }
        }

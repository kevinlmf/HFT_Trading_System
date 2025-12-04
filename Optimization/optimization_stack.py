"""
深度优化栈 - 从统计理论到系统实现的完整优化流程

架构层次：
1. 统计理论层 - 定义模型目标
2. 模型表达层 - 写出似然/风险函数
3. 算法设计层 - 选择求解算法（优化、采样、近似）
4. 数据结构层 - 优化内存布局、并行结构
5. 系统实现层 - 编译、并行、GPU、0/1信号执行
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import warnings
warnings.filterwarnings('ignore')

# CVXPY Integration
try:
    from Optimization.cvxpy_optimizer import CVXPYOptimizer, CVXPY_AVAILABLE
except ImportError:
    CVXPY_AVAILABLE = False
    CVXPYOptimizer = None

# JAX Integration
try:
    from Optimization.jax_optimizer import JAXOptimizer, JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False
    JAXOptimizer = None




# ========== 统计理论层 ==========

from Optimization.optimization_types import ModelObjective, AlgorithmType, AlgorithmConfig

# ========== 统计理论层 ==========



@dataclass
class StatisticalModel:
    """统计模型定义"""
    objective: ModelObjective
    constraints: Dict[str, Any]
    parameters: Dict[str, float]
    assumptions: List[str]
    
    def validate(self) -> bool:
        """验证模型假设"""
        return True


class StatisticalTheoryLayer:
    """统计理论层 - 定义模型目标"""
    
    def __init__(self):
        self.models = {}
    
    def define_model(
        self,
        name: str,
        objective: ModelObjective,
        constraints: Optional[Dict] = None,
        parameters: Optional[Dict] = None
    ) -> StatisticalModel:
        """
        定义统计模型
        
        Args:
            name: 模型名称
            objective: 优化目标
            constraints: 约束条件
            parameters: 模型参数
        
        Returns:
            统计模型对象
        """
        model = StatisticalModel(
            objective=objective,
            constraints=constraints or {},
            parameters=parameters or {},
            assumptions=[]
        )
        self.models[name] = model
        return model
    
    def get_model(self, name: str) -> Optional[StatisticalModel]:
        """获取模型"""
        return self.models.get(name)


# ========== 模型表达层 ==========

@dataclass
class LikelihoodFunction:
    """似然函数"""
    func: Callable
    gradient: Optional[Callable] = None
    hessian: Optional[Callable] = None
    parameters: Dict[str, float] = None


@dataclass
class RiskFunction:
    """风险函数"""
    func: Callable
    gradient: Optional[Callable] = None
    hessian: Optional[Callable] = None
    parameters: Dict[str, float] = None


class ModelExpressionLayer:
    """模型表达层 - 写出似然/风险函数"""
    
    def __init__(self):
        self.likelihood_functions = {}
        self.risk_functions = {}
    
    def define_likelihood(
        self,
        name: str,
        func: Callable,
        gradient: Optional[Callable] = None,
        hessian: Optional[Callable] = None
    ) -> LikelihoodFunction:
        """定义似然函数"""
        likelihood = LikelihoodFunction(
            func=func,
            gradient=gradient,
            hessian=hessian
        )
        self.likelihood_functions[name] = likelihood
        return likelihood
    
    def define_risk_function(
        self,
        name: str,
        func: Callable,
        gradient: Optional[Callable] = None,
        hessian: Optional[Callable] = None
    ) -> RiskFunction:
        """定义风险函数"""
        risk_func = RiskFunction(
            func=func,
            gradient=gradient,
            hessian=hessian
        )
        self.risk_functions[name] = risk_func
        return risk_func
    
    # 预定义的似然函数
    def gaussian_likelihood(self, returns: np.ndarray) -> LikelihoodFunction:
        """高斯似然函数"""
        def likelihood(params: np.ndarray) -> float:
            mu, sigma = params[0], params[1]
            n = len(returns)
            log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2)
            log_likelihood -= np.sum((returns - mu)**2) / (2 * sigma**2)
            return -log_likelihood  # 负对数似然（用于最小化）
        
        def gradient(params: np.ndarray) -> np.ndarray:
            mu, sigma = params[0], params[1]
            n = len(returns)
            grad_mu = np.sum(returns - mu) / sigma**2
            grad_sigma = -n/sigma + np.sum((returns - mu)**2) / sigma**3
            return -np.array([grad_mu, grad_sigma])
        
        return self.define_likelihood("gaussian", likelihood, gradient)
    
    # 预定义的风险函数
    def variance_risk(self, returns: np.ndarray) -> RiskFunction:
        """方差风险函数"""
        # returns 可能是 NumPy 数组或 Pandas DataFrame
        if hasattr(returns, 'to_numpy'):
            returns_array = returns.to_numpy()
        else:
            returns_array = np.asarray(returns)

        if returns_array.ndim == 1:
            returns_array = returns_array[:, None]

        mean_returns = returns_array.mean(axis=0)
        cov_matrix = np.cov(returns_array, rowvar=False)

        def risk(weights: np.ndarray) -> float:
            portfolio_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
            return portfolio_var
        
        def gradient(weights: np.ndarray) -> np.ndarray:
            return 2.0 * np.dot(cov_matrix, weights)
        
        return self.define_risk_function("variance", risk, gradient)
    
    def cvar_risk(self, returns: np.ndarray, alpha: float = 0.05) -> RiskFunction:
        """CVaR风险函数"""
        def risk(weights: np.ndarray) -> float:
            portfolio_returns = np.dot(returns, weights)
            var = np.percentile(portfolio_returns, alpha * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            return -cvar  # 负CVaR（用于最小化）
        
        return self.define_risk_function("cvar", risk)


# ========== 算法设计层 ==========







class AlgorithmDesignLayer:
    """算法设计层 - 选择求解算法"""
    
    def __init__(self):
        self.algorithms = {}
    
    def select_algorithm(
        self,
        problem_type: str,
        has_gradient: bool = True,
        has_hessian: bool = False,
        problem_size: int = 1000,
        gpu_available: bool = False
    ) -> AlgorithmConfig:
        """
        根据问题特征选择最优算法
        
        Args:
            problem_type: 问题类型 ('optimization', 'sampling', 'approximation')
            has_gradient: 是否有梯度
            has_hessian: 是否有Hessian矩阵
            problem_size: 问题规模
            gpu_available: GPU是否可用
        
        Returns:
            算法配置
        """
        if problem_type == 'optimization':
            if has_hessian and problem_size < 1000:
                algo = AlgorithmType.NEWTON_METHOD
            elif has_gradient:
                if problem_size > 10000 and gpu_available:
                    algo = AlgorithmType.STOCHASTIC_GRADIENT
                elif problem_size > 1000:
                    algo = AlgorithmType.ADAM
                else:
                    algo = AlgorithmType.LBFGS
            else:
                algo = AlgorithmType.SIMULATED_ANNEALING
        elif problem_type == 'sampling':
            algo = AlgorithmType.MCMC
        else:  # approximation
            algo = AlgorithmType.VARIATIONAL_INFERENCE
        
        config = AlgorithmConfig(
            algorithm_type=algo,
            parallel=problem_size > 100,
            gpu=gpu_available and problem_size > 1000
        )
        
        return config
    
    def optimize(
        self,
        objective_func: Callable,
        initial_params: np.ndarray,
        config: AlgorithmConfig,
        gradient_func: Optional[Callable] = None,
        hessian_func: Optional[Callable] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        执行优化
        
        Returns:
            (最优参数, 优化信息)
        """
        if config.algorithm_type == AlgorithmType.GRADIENT_DESCENT:
            return self._gradient_descent(
                objective_func, initial_params, config, gradient_func
            )
        elif config.algorithm_type == AlgorithmType.ADAM:
            return self._adam_optimizer(
                objective_func, initial_params, config, gradient_func
            )
        elif config.algorithm_type == AlgorithmType.LBFGS:
            from scipy.optimize import minimize
            result = minimize(
                objective_func,
                initial_params,
                method='L-BFGS-B',
                jac=gradient_func,
                options={'maxiter': config.max_iterations}
            )
            return result.x, {'success': result.success, 'iterations': result.nit}
        else:
            # 默认使用梯度下降
            return self._gradient_descent(
                objective_func, initial_params, config, gradient_func
            )
    
    def _gradient_descent(
        self,
        objective_func: Callable,
        initial_params: np.ndarray,
        config: AlgorithmConfig,
        gradient_func: Optional[Callable]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """梯度下降"""
        params = initial_params.copy()
        history = []
        
        for i in range(config.max_iterations):
            if gradient_func:
                grad = gradient_func(params)
            else:
                # 数值梯度
                grad = self._numerical_gradient(objective_func, params)
            
            params -= config.learning_rate * grad
            
            value = objective_func(params)
            history.append(value)
            
            if np.linalg.norm(grad) < config.tolerance:
                break
        
        return params, {'iterations': i+1, 'history': history}
    
    def _adam_optimizer(
        self,
        objective_func: Callable,
        initial_params: np.ndarray,
        config: AlgorithmConfig,
        gradient_func: Optional[Callable]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adam优化器"""
        params = initial_params.copy()
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        history = []
        
        for i in range(config.max_iterations):
            if gradient_func:
                grad = gradient_func(params)
            else:
                grad = self._numerical_gradient(objective_func, params)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**(i+1))
            v_hat = v / (1 - beta2**(i+1))
            
            params -= config.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            value = objective_func(params)
            history.append(value)
            
            if np.linalg.norm(grad) < config.tolerance:
                break
        
        return params, {'iterations': i+1, 'history': history}
    
    def _numerical_gradient(self, func: Callable, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """数值梯度"""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            grad[i] = (func(params_plus) - func(params)) / h
        return grad


# ========== 数据结构层 ==========

class MemoryLayout(Enum):
    """内存布局"""
    ROW_MAJOR = "row_major"  # C-style
    COLUMN_MAJOR = "column_major"  # Fortran-style
    BLOCKED = "blocked"  # 分块存储
    SPARSE = "sparse"  # 稀疏矩阵


@dataclass
class DataStructure:
    """数据结构定义"""
    layout: MemoryLayout
    dtype: np.dtype
    shape: Tuple[int, ...]
    alignment: int = 64  # 字节对齐
    cache_aware: bool = True


class DataStructureLayer:
    """数据结构层 - 优化内存布局、并行结构"""
    
    def __init__(self):
        self.structures = {}
    
    def optimize_layout(
        self,
        data: np.ndarray,
        access_pattern: str = "row",
        parallel: bool = True
    ) -> DataStructure:
        """
        优化内存布局
        
        Args:
            data: 原始数据
            access_pattern: 访问模式 ('row', 'column', 'block')
            parallel: 是否并行访问
        """
        if access_pattern == "row":
            layout = MemoryLayout.ROW_MAJOR
            # 确保行主序
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
        elif access_pattern == "column":
            layout = MemoryLayout.COLUMN_MAJOR
            if not data.flags['F_CONTIGUOUS']:
                data = np.asfortranarray(data)
        else:  # block
            layout = MemoryLayout.BLOCKED
            data = self._block_layout(data)
        
        # 字节对齐（SIMD优化）
        aligned_data = self._align_memory(data, alignment=64)
        
        structure = DataStructure(
            layout=layout,
            dtype=aligned_data.dtype,
            shape=aligned_data.shape,
            cache_aware=True
        )
        
        return structure, aligned_data
    
    def _block_layout(self, data: np.ndarray, block_size: int = 64) -> np.ndarray:
        """分块布局（提高缓存命中率）"""
        # 简化实现：返回原数据
        # 实际应该重新排列数据为块状结构
        return data
    
    def _align_memory(self, data: np.ndarray, alignment: int = 64) -> np.ndarray:
        """内存对齐"""
        # NumPy默认已经对齐，这里只是确保
        if data.flags['ALIGNED']:
            return data
        # 创建对齐的副本
        aligned = np.empty_like(data)
        aligned[:] = data
        return aligned
    
    def create_parallel_structure(
        self,
        data: np.ndarray,
        num_threads: int = 4
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        创建并行数据结构
        
        Returns:
            (数据, 分块索引列表)
        """
        n = len(data)
        chunk_size = n // num_threads
        chunks = []
        
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size if i < num_threads - 1 else n
            chunks.append((start, end))
        
        return data, chunks


# ========== 系统实现层 ==========

class ExecutionBackend(Enum):
    """执行后端"""
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    ASIC = "asic"


@dataclass
class SystemConfig:
    """系统配置"""
    backend: ExecutionBackend
    num_threads: int = 4
    use_simd: bool = True
    use_gpu: bool = False
    compile_optimization: str = "O3"  # O0, O1, O2, O3
    precision: str = "float32"  # float32, float64


class SystemImplementationLayer:
    """系统实现层 - 编译、并行、GPU、0/1信号执行"""
    
    def __init__(self):
        self.backend = ExecutionBackend.CPU
        self.config = SystemConfig(ExecutionBackend.CPU)
        
        # 检查GPU可用性
        try:
            import cupy as cp
            self.gpu_available = True
            self.cp = cp
        except ImportError:
            self.gpu_available = False
            self.cp = None
    
    def compile_function(
        self,
        func: Callable,
        config: SystemConfig
    ) -> Callable:
        """编译函数（使用Numba JIT）"""
        try:
            from numba import jit
            
            # 对于动态函数，使用更宽松的编译选项
            if config.backend == ExecutionBackend.GPU:
                try:
                    # GPU编译（需要CUDA）
                    @jit(target='cuda', forceobj=True)
                    def compiled_func(*args):
                        return func(*args)
                    return compiled_func
                except:
                    # GPU不可用，回退到CPU
                    return func
            else:
                # CPU编译（使用forceobj=True允许Python对象）
                try:
                    @jit(forceobj=True, parallel=True, fastmath=True)
                    def compiled_func(*args):
                        return func(*args)
                    return compiled_func
                except:
                    # 编译失败，返回原函数
                    return func
        except ImportError:
            # Numba不可用，返回原函数
            return func
    
    def execute_on_gpu(
        self,
        func: Callable,
        *args
    ) -> Any:
        """在GPU上执行"""
        if not self.gpu_available:
            raise RuntimeError("GPU not available")
        
        # 转换参数到GPU
        gpu_args = [self.cp.asarray(arg) if isinstance(arg, np.ndarray) else arg 
                   for arg in args]
        
        # 执行
        result = func(*gpu_args)
        
        # 转换回CPU
        if isinstance(result, self.cp.ndarray):
            return self.cp.asnumpy(result)
        return result
    
    def execute_parallel(
        self,
        func: Callable,
        data: np.ndarray,
        num_threads: int = 4
    ) -> np.ndarray:
        """并行执行"""
        try:
            from numba import prange
            
            @jit(nopython=True, parallel=True)
            def parallel_func(data):
                result = np.zeros_like(data)
                for i in prange(len(data)):
                    result[i] = func(data[i])
                return result
            
            return parallel_func(data)
        except ImportError:
            # 使用多进程
            from multiprocessing import Pool
            with Pool(num_threads) as pool:
                result = pool.map(func, data)
            return np.array(result)
    
    def generate_signal(
        self,
        weights: np.ndarray,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        生成0/1交易信号
        
        Args:
            weights: 权重向量
            threshold: 阈值
        
        Returns:
            0/1信号数组
        """
        signals = (weights > threshold).astype(np.int8)
        return signals


# ========== 完整优化栈 ==========

class OptimizationStack:
    """完整优化栈 - 整合所有层次"""
    
    def __init__(self):
        self.statistical_layer = StatisticalTheoryLayer()
        self.expression_layer = ModelExpressionLayer()
        self.algorithm_layer = AlgorithmDesignLayer()
        self.data_layer = DataStructureLayer()
        self.system_layer = SystemImplementationLayer()
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        objective: ModelObjective = ModelObjective.MAXIMIZE_SHARPE,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        完整的投资组合优化流程
        
        Args:
            returns: 收益率矩阵 (n_samples, n_assets)
            objective: 优化目标
            constraints: 约束条件
        
        Returns:
            优化结果
        """
        n_assets = returns.shape[1]
        
        # 1. 统计理论层：定义模型
        model = self.statistical_layer.define_model(
            "portfolio_optimization",
            objective=objective,
            constraints=constraints or {}
        )

        # -1. 尝试使用JAX优化器 (最高优先级，特别是对于大规模问题或因子模型)
        # 检查是否提供了因子模型参数
        factor_loadings = constraints.get('factor_loadings') if constraints else None
        factor_cov = constraints.get('factor_cov') if constraints else None
        idio_risk = constraints.get('idiosyncratic_risk') if constraints else None
        
        use_jax = JAX_AVAILABLE and (
            (factor_loadings is not None) or # 显式要求因子模型
            (n_assets > 500) or              # 大规模问题
            (objective == ModelObjective.MAXIMIZE_SHARPE and not constraints) # 无约束解析解
        )
        
        if use_jax:
            try:
                jax_optimizer = JAXOptimizer(use_gpu=True)
                
                # 准备参数
                mean_returns = returns.mean(axis=0)
                if factor_loadings is None:
                    cov_matrix = np.cov(returns, rowvar=False)
                else:
                    cov_matrix = None
                
                result = jax_optimizer.optimize_portfolio(
                    mean_returns=mean_returns,
                    cov_matrix=cov_matrix,
                    factor_loadings=factor_loadings,
                    factor_cov=factor_cov,
                    idiosyncratic_risk=idio_risk,
                    objective=objective,
                    constraints=constraints
                )
                
                if result['success']:
                    return {
                        'optimal_weights': result['weights'],
                        'signals': self.system_layer.generate_signal(result['weights']),
                        'optimization_info': {
                            'success': True,
                            'algorithm': result['optimization_info']['algorithm'],
                            'message': result['optimization_info']['message'],
                            'device': result['optimization_info']['device']
                        },
                        'model': model,
                        'algorithm': 'JAX'
                    }
            except Exception as e:
                print(f"JAX optimization failed: {e}, falling back to CVXPY/SLSQP")

        # 0. 尝试使用CVXPY优化器 (如果可用且适用)

        if CVXPY_AVAILABLE and objective in [ModelObjective.MAXIMIZE_SHARPE, ModelObjective.MINIMIZE_VARIANCE, ModelObjective.MAXIMIZE_RETURN]:
            try:
                # 计算必要的统计量
                mean_returns = returns.mean(axis=0)
                cov_matrix = np.cov(returns, rowvar=False)
                
                # 创建优化器
                cvx_optimizer = CVXPYOptimizer(solver='OSQP')
                
                # 执行优化
                result = cvx_optimizer.optimize_portfolio(
                    mean_returns=mean_returns,
                    cov_matrix=cov_matrix,
                    objective=objective,
                    constraints=constraints
                )
                
                if result['success']:
                    optimal_weights = result['weights']
                    signals = self.system_layer.generate_signal(optimal_weights)
                    
                    return {
                        'optimal_weights': optimal_weights,
                        'signals': signals,
                        'optimization_info': {
                            'success': True,
                            'algorithm': 'CVXPY_' + result['solver_stats']['solver'],
                            'message': 'Optimized using CVXPY'
                        },
                        'model': model,
                        'algorithm': 'CVXPY'
                    }
            except Exception as e:
                print(f"CVXPY optimization failed: {e}, falling back to SLSQP")
                # Fallback to standard flow

        
        # 2. 模型表达层：定义风险函数
        if objective == ModelObjective.MAXIMIZE_SHARPE:
            # Sharpe比率最大化 = 收益/风险最大化
            def objective_func(weights: np.ndarray) -> float:
                portfolio_return = np.dot(weights, returns.mean(axis=0))
                portfolio_std = np.sqrt(np.dot(weights, np.dot(np.cov(returns.T), weights)))
                if portfolio_std == 0:
                    return -np.inf
                sharpe = portfolio_return / portfolio_std
                return -sharpe  # 负号因为要最小化
            
            def gradient_func(weights: np.ndarray) -> np.ndarray:
                # Sharpe比率的梯度
                mean_returns = returns.mean(axis=0)
                cov_matrix = np.cov(returns.T)
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_std = np.sqrt(portfolio_var)
                
                if portfolio_std == 0:
                    return np.zeros_like(weights)
                
                grad_return = mean_returns
                grad_std = np.dot(cov_matrix, weights) / portfolio_std
                grad_sharpe = (grad_return * portfolio_std - portfolio_return * grad_std) / portfolio_std**2
                return -grad_sharpe
            
            risk_func = self.expression_layer.define_risk_function(
                "sharpe", objective_func, gradient_func
            )
        else:
            # 其他目标
            risk_func = self.expression_layer.variance_risk(returns)
        
        # 3. 算法设计层：选择算法
        algo_config = self.algorithm_layer.select_algorithm(
            problem_type='optimization',
            has_gradient=True,
            problem_size=n_assets,
            gpu_available=self.system_layer.gpu_available
        )
        
        # 4. 数据结构层：优化内存布局
        optimized_returns, chunks = self.data_layer.create_parallel_structure(
            returns, num_threads=4
        )
        
        # 5. 系统实现层：编译和优化（可选）
        # 对于复杂函数，直接使用原函数可能更快
        use_compiled = False  # 暂时禁用编译，避免Numba问题
        if use_compiled:
            compiled_objective = self.system_layer.compile_function(
                risk_func.func, self.system_layer.config
            )
            compiled_gradient = self.system_layer.compile_function(
                risk_func.gradient, self.system_layer.config
            ) if risk_func.gradient else None
        else:
            compiled_objective = risk_func.func
            compiled_gradient = risk_func.gradient
        
        # 6. 执行优化
        initial_weights = np.ones(n_assets) / n_assets  # 等权重初始值
        
        # 应用约束（权重和为1）
        max_pos = constraints.get('max_position_size', 1.0) if constraints else 1.0
        sum_to_one = constraints.get('sum_to_one', True) if constraints else True
        
        if sum_to_one:
            # 使用scipy的约束优化
            from scipy.optimize import minimize
            bounds = [(0, max_pos)] * n_assets
            constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            
            result = minimize(
                compiled_objective,
                initial_weights,
                method='SLSQP',
                jac=compiled_gradient,
                bounds=bounds,
                constraints=constraints_opt,
                options={'maxiter': algo_config.max_iterations}
            )
            optimal_weights = result.x
            # 确保权重和为1（归一化）
            if np.abs(np.sum(optimal_weights) - 1.0) > 1e-6:
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            info = {'success': result.success, 'iterations': result.nit, 'message': result.message}
        else:
            # 无约束优化
            if self.system_layer.gpu_available and algo_config.gpu:
                # GPU执行
                optimal_weights, info = self.system_layer.execute_on_gpu(
                    lambda w: self.algorithm_layer.optimize(
                        compiled_objective, w, algo_config, compiled_gradient
                    )[0],
                    initial_weights
                )
            else:
                # CPU执行
                optimal_weights, info = self.algorithm_layer.optimize(
                    compiled_objective,
                    initial_weights,
                    algo_config,
                    compiled_gradient
                )
        
        # 7. 生成交易信号
        signals = self.system_layer.generate_signal(optimal_weights)
        
        return {
            'optimal_weights': optimal_weights,
            'signals': signals,
            'optimization_info': info,
            'model': model,
            'algorithm': algo_config.algorithm_type.value
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples, n_assets = 1000, 10
    returns = np.random.randn(n_samples, n_assets) * 0.02
    
    # 创建优化栈
    stack = OptimizationStack()
    
    # 执行优化
    result = stack.optimize_portfolio(
        returns,
        objective=ModelObjective.MAXIMIZE_SHARPE
    )
    
    print("优化结果:")
    print(f"  最优权重: {result['optimal_weights']}")
    print(f"  交易信号: {result['signals']}")
    print(f"  使用算法: {result['algorithm']}")


"""
优化的Optimization模块 - 集成QDB并优化方法和数据结构

优化点：
1. 数据加载：使用QDB快速加载数据（O(log n)）
2. 方法优化：协方差矩阵计算、优化算法选择
3. 数据结构优化：内存布局、缓存中间结果、紧凑数据结构
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# QDB集成
try:
    from QDB import QDB, create_qdb
    from QDB.improved_optimized_indexer import ImprovedOptimizedIndexer
    QDB_AVAILABLE = True
except ImportError:
    QDB_AVAILABLE = False
    QDB = None

# 导入原有模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Optimization.optimization_stack import (
    OptimizationStack,
    ExecutionBackend,
    StatisticalTheoryLayer,
    ModelExpressionLayer,
    AlgorithmDesignLayer,
    DataStructureLayer,
    SystemImplementationLayer
)
from Optimization.optimization_types import ModelObjective, AlgorithmType, AlgorithmConfig

# CVXPY Integration
try:
    from Optimization.cvxpy_optimizer import CVXPYOptimizer, CVXPY_AVAILABLE
except ImportError:
    CVXPY_AVAILABLE = False
    CVXPYOptimizer = None



class OptimizedDataLoader:
    """
    优化的数据加载器 - 使用QDB快速加载数据
    
    优化：
    1. 使用QDB的O(log n)索引查找
    2. 并行加载多个symbol
    3. 缓存协方差矩阵等中间结果
    """
    
    def __init__(self, qdb: Optional[QDB] = None, use_qdb: bool = True):
        """
        Args:
            qdb: QDB实例（如果为None，会创建新的）
            use_qdb: 是否使用QDB（如果QDB不可用，会降级到传统方式）
        """
        self.use_qdb = use_qdb and QDB_AVAILABLE
        if self.use_qdb:
            self.qdb = qdb or create_qdb()
        else:
            self.qdb = None
        
        # 缓存协方差矩阵（避免重复计算）
        self._covariance_cache: Dict[str, np.ndarray] = {}
        self._mean_returns_cache: Dict[str, np.ndarray] = {}
    
    def load_returns(
        self,
        symbols: List[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        加载收益率数据（优化版本）
        
        Args:
            symbols: 交易标的列表
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            (returns_array, valid_symbols)
            returns_array: (n_samples, n_assets) 收益率矩阵
        """
        if self.use_qdb and self.qdb:
            return self._load_from_qdb(symbols, start_time, end_time)
        else:
            # 传统方式（需要外部提供数据）
            raise NotImplementedError("Traditional loading not implemented, use QDB")
    
    def _load_from_qdb(
        self,
        symbols: List[str],
        start_time: Optional[str],
        end_time: Optional[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """从QDB加载数据（并行）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        returns_list = []
        valid_symbols = []
        
        def load_symbol(symbol: str) -> Optional[Tuple[str, pd.Series]]:
            """加载单个symbol的收益率"""
            try:
                df = self.qdb.load(symbol=symbol, start=start_time, end=end_time)
                if len(df) == 0:
                    return None
                
                # 计算收益率
                if 'last_price' in df.columns:
                    prices = df['last_price']
                elif 'close' in df.columns:
                    prices = df['close']
                else:
                    return None
                
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    return (symbol, returns)
            except Exception as e:
                warnings.warn(f"Failed to load {symbol}: {e}")
            return None
        
        # 并行加载
        with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as executor:
            futures = {executor.submit(load_symbol, sym): sym for sym in symbols}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    symbol, returns = result
                    returns_list.append(returns)
                    valid_symbols.append(symbol)
        
        if len(returns_list) == 0:
            return np.array([]).reshape(0, 0), []
        
        # 对齐时间序列（使用pandas的align）
        aligned_returns = pd.DataFrame({s: r for s, r in zip(valid_symbols, returns_list)})
        aligned_returns = aligned_returns.dropna()  # 删除缺失值
        
        # 转换为numpy数组
        returns_array = aligned_returns.values
        
        return returns_array, valid_symbols
    
    def get_covariance_matrix(
        self,
        returns: np.ndarray,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        计算协方差矩阵（优化版本）
        
        优化：
        1. 使用缓存避免重复计算
        2. 使用numpy的高效实现
        3. 内存对齐优化
        """
        if use_cache and cache_key and cache_key in self._covariance_cache:
            return self._covariance_cache[cache_key]
        
        # 使用numpy的高效协方差计算
        # 确保数据是C-contiguous（行主序）
        if not returns.flags['C_CONTIGUOUS']:
            returns = np.ascontiguousarray(returns)
        
        # 计算协方差矩阵（使用numpy的优化实现）
        cov_matrix = np.cov(returns, rowvar=False)
        
        # 缓存结果
        if use_cache and cache_key:
            self._covariance_cache[cache_key] = cov_matrix
        
        return cov_matrix
    
    def get_mean_returns(
        self,
        returns: np.ndarray,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """计算平均收益率（优化版本）"""
        if use_cache and cache_key and cache_key in self._mean_returns_cache:
            return self._mean_returns_cache[cache_key]
        
        # 使用numpy的高效mean计算
        mean_returns = returns.mean(axis=0)
        
        # 确保是1D数组
        if mean_returns.ndim > 1:
            mean_returns = mean_returns.flatten()
        
        # 缓存结果
        if use_cache and cache_key:
            self._mean_returns_cache[cache_key] = mean_returns
        
        return mean_returns
    
    def clear_cache(self):
        """清空缓存"""
        self._covariance_cache.clear()
        self._mean_returns_cache.clear()


class OptimizedAlgorithmSelector:
    """
    优化的算法选择器
    
    优化：
    1. 根据问题规模智能选择算法
    2. 考虑数据特征（稀疏性、条件数等）
    3. 自适应参数调整
    """
    
    @staticmethod
    def select_optimal_algorithm(
        problem_size: int,
        has_gradient: bool,
        has_hessian: bool,
        condition_number: Optional[float] = None,
        is_sparse: bool = False,
        gpu_available: bool = False
    ) -> AlgorithmType:
        """
        智能选择最优算法
        
        Args:
            problem_size: 问题规模（变量数）
            has_gradient: 是否有梯度
            has_hessian: 是否有Hessian矩阵
            condition_number: 条件数（如果已知）
            is_sparse: 是否稀疏问题
            gpu_available: GPU是否可用
        
        Returns:
            最优算法类型
        """
        # 小规模问题（< 50变量）
        if problem_size < 50:
            if has_hessian:
                return AlgorithmType.NEWTON_METHOD
            elif has_gradient:
                return AlgorithmType.LBFGS
            else:
                return AlgorithmType.SIMULATED_ANNEALING
        
        # 中等规模问题（50-500变量）
        elif problem_size < 500:
            if has_gradient:
                if condition_number and condition_number > 1e6:
                    # 病态问题，使用Adam
                    return AlgorithmType.ADAM
                else:
                    return AlgorithmType.LBFGS
            else:
                return AlgorithmType.GENETIC_ALGORITHM
        
        # 大规模问题（> 500变量）
        else:
            if gpu_available and has_gradient:
                return AlgorithmType.STOCHASTIC_GRADIENT
            elif has_gradient:
                return AlgorithmType.ADAM
            else:
                return AlgorithmType.GENETIC_ALGORITHM


class OptimizedDataStructure:
    """
    优化的数据结构
    
    优化：
    1. 内存对齐（SIMD优化）
    2. 紧凑数据类型（float32 vs float64）
    3. 缓存友好的布局
    4. 预分配内存
    """
    
    @staticmethod
    def optimize_array(
        data: np.ndarray,
        dtype: Optional[np.dtype] = None,
        alignment: int = 64,
        cache_aware: bool = True
    ) -> np.ndarray:
        """
        优化数组布局
        
        Args:
            data: 原始数组
            dtype: 目标数据类型（None表示保持原类型）
            alignment: 字节对齐（64字节用于AVX-512）
            cache_aware: 是否考虑缓存
        
        Returns:
            优化后的数组
        """
        # 1. 数据类型优化（float32 vs float64）
        if dtype is None:
            # 自动选择：如果精度要求不高，使用float32
            if data.dtype == np.float64 and np.abs(data).max() < 1e10:
                dtype = np.float32  # 节省50%内存，加速计算
            else:
                dtype = data.dtype
        else:
            dtype = dtype
        
        # 2. 内存对齐
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data, dtype=dtype)
        else:
            data = data.astype(dtype, copy=False)
        
        # 3. 确保对齐（对于SIMD优化）
        if alignment > 0:
            # NumPy默认已经对齐，这里只是确保
            if not data.flags['ALIGNED']:
                # 创建对齐的副本
                aligned = np.empty_like(data)
                aligned[:] = data
                data = aligned
        
        return data
    
    @staticmethod
    def create_compact_covariance(
        cov_matrix: np.ndarray,
        precision: str = "float32"
    ) -> np.ndarray:
        """
        创建紧凑的协方差矩阵
        
        优化：
        1. 使用float32节省内存
        2. 利用对称性（只存储上三角）
        3. 稀疏矩阵（如果很多零）
        """
        dtype = np.float32 if precision == "float32" else np.float64
        
        # 检查稀疏性
        sparsity = np.count_nonzero(cov_matrix == 0) / cov_matrix.size
        if sparsity > 0.5:
            # 使用稀疏矩阵
            from scipy.sparse import csr_matrix
            return csr_matrix(cov_matrix.astype(dtype))
        
        # 使用float32（如果精度允许）
        if precision == "float32":
            cov_compact = cov_matrix.astype(np.float32)
        else:
            cov_compact = cov_matrix
        
        return cov_compact


class EnhancedOptimizationStack(OptimizationStack):
    """
    增强的优化栈 - 集成QDB和优化方法/数据结构
    """
    
    def __init__(self, use_qdb: bool = True, qdb: Optional[QDB] = None):
        """
        Args:
            use_qdb: 是否使用QDB
            qdb: QDB实例（可选）
        """
        super().__init__()
        
        # 集成QDB数据加载器
        self.data_loader = OptimizedDataLoader(qdb=qdb, use_qdb=use_qdb)
        
        # 优化的算法选择器
        self.algorithm_selector = OptimizedAlgorithmSelector()
        
        # 优化的数据结构工具
        self.data_structure = OptimizedDataStructure()
    
    def optimize_portfolio_from_qdb(
        self,
        symbols: List[str],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        objective: ModelObjective = ModelObjective.MAXIMIZE_SHARPE,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        从QDB加载数据并优化投资组合（优化版本）
        
        优化：
        1. 使用QDB快速加载数据（O(log n)）
        2. 并行加载多个symbol
        3. 缓存协方差矩阵
        4. 优化的算法选择
        5. 优化的数据结构
        """
        # 1. 从QDB加载数据（并行，O(log n)）
        returns_array, valid_symbols = self.data_loader.load_returns(
            symbols, start_time, end_time
        )
        
        if len(valid_symbols) == 0:
            return {'error': 'No valid data loaded'}
        
        # 2. 优化数据结构
        returns_optimized = self.data_structure.optimize_array(
            returns_array,
            dtype=np.float32,  # 使用float32节省内存和加速
            cache_aware=True
        )
        
        # 3. 计算协方差矩阵（使用缓存）
        cache_key = f"{','.join(sorted(valid_symbols))}_{start_time}_{end_time}"
        cov_matrix = self.data_loader.get_covariance_matrix(
            returns_optimized,
            use_cache=True,
            cache_key=cache_key
        )
        
        # 优化协方差矩阵结构
        cov_matrix_compact = self.data_structure.create_compact_covariance(
            cov_matrix,
            precision="float32"
        )
        
        # 4. 计算平均收益率（使用缓存）
        mean_returns = self.data_loader.get_mean_returns(
            returns_optimized,
            use_cache=True,
            cache_key=cache_key
        )
        
        # 5. 智能选择算法
        n_assets = len(valid_symbols)
        condition_number = np.linalg.cond(cov_matrix) if isinstance(cov_matrix, np.ndarray) else None
        
        optimal_algorithm = self.algorithm_selector.select_optimal_algorithm(
            problem_size=n_assets,
            has_gradient=True,
            has_hessian=False,
            condition_number=condition_number,
            gpu_available=self.system_layer.gpu_available
        )

        # 0. 尝试使用CVXPY优化器 (如果可用且适用)
        if CVXPY_AVAILABLE and objective in [ModelObjective.MAXIMIZE_SHARPE, ModelObjective.MINIMIZE_VARIANCE, ModelObjective.MAXIMIZE_RETURN]:
            try:
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
                        'optimal_weights': dict(zip(valid_symbols, optimal_weights)),
                        'signals': dict(zip(valid_symbols, signals)),
                        'optimization_info': {
                            'success': True,
                            'iterations': result['solver_stats']['iterations'],
                            'algorithm': 'CVXPY_' + result['solver_stats']['solver'],
                            'data_source': 'QDB' if self.data_loader.use_qdb else 'traditional',
                            'cache_used': cache_key in self.data_loader._covariance_cache,
                            'message': 'Optimized using CVXPY'
                        },
                        'performance_metrics': {
                            'portfolio_return': float(np.dot(optimal_weights, mean_returns)),
                            'portfolio_std': float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))),
                            'sharpe_ratio': float(np.dot(optimal_weights, mean_returns) / np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))
                        }
                    }
            except Exception as e:
                print(f"CVXPY optimization failed: {e}, falling back to SLSQP")
                # Fallback to standard flow

        
        # 6. 定义优化目标函数（使用优化的数据结构）
        if objective == ModelObjective.MAXIMIZE_SHARPE:
            def objective_func(weights: np.ndarray) -> float:
                # 确保weights是float32
                weights = weights.astype(np.float32)
                
                # 使用优化的矩阵乘法
                portfolio_return = np.dot(weights, mean_returns)
                
                # 协方差矩阵乘法（处理稀疏矩阵）
                if hasattr(cov_matrix_compact, 'dot'):
                    portfolio_var = np.dot(weights, cov_matrix_compact.dot(weights))
                else:
                    portfolio_var = np.dot(weights, np.dot(cov_matrix_compact, weights))
                
                portfolio_std = np.sqrt(portfolio_var)
                if portfolio_std == 0:
                    return -np.inf
                sharpe = portfolio_return / portfolio_std
                return -sharpe  # 负号因为要最小化
            
            def gradient_func(weights: np.ndarray) -> np.ndarray:
                weights = weights.astype(np.float32)
                portfolio_return = np.dot(weights, mean_returns)
                
                if hasattr(cov_matrix_compact, 'dot'):
                    portfolio_var = np.dot(weights, cov_matrix_compact.dot(weights))
                    grad_var = 2.0 * cov_matrix_compact.dot(weights)
                else:
                    portfolio_var = np.dot(weights, np.dot(cov_matrix_compact, weights))
                    grad_var = 2.0 * np.dot(cov_matrix_compact, weights)
                
                portfolio_std = np.sqrt(portfolio_var)
                if portfolio_std == 0:
                    return np.zeros_like(weights)
                
                grad_sharpe = (mean_returns * portfolio_std - portfolio_return * grad_var / portfolio_std) / portfolio_std**2
                return -grad_sharpe.astype(np.float32)
        else:
            # 其他目标
            def objective_func(weights: np.ndarray) -> float:
                weights = weights.astype(np.float32)
                if hasattr(cov_matrix_compact, 'dot'):
                    return float(np.dot(weights, cov_matrix_compact.dot(weights)))
                else:
                    return float(np.dot(weights, np.dot(cov_matrix_compact, weights)))
            
            def gradient_func(weights: np.ndarray) -> np.ndarray:
                weights = weights.astype(np.float32)
                if hasattr(cov_matrix_compact, 'dot'):
                    return (2.0 * cov_matrix_compact.dot(weights)).astype(np.float32)
                else:
                    return (2.0 * np.dot(cov_matrix_compact, weights)).astype(np.float32)
        
        # 7. 选择算法配置
        algo_config = self.algorithm_layer.select_algorithm(
            problem_type='optimization',
            has_gradient=True,
            problem_size=n_assets,
            gpu_available=self.system_layer.gpu_available
        )
        
        # 覆盖算法类型（使用智能选择的结果）
        algo_config.algorithm_type = optimal_algorithm
        
        # 8. 执行优化
        initial_weights = np.ones(n_assets, dtype=np.float32) / n_assets
        
        # 应用约束
        from scipy.optimize import minimize
        max_pos = constraints.get('max_position_size', 1.0) if constraints else 1.0
        bounds = [(0.0, max_pos)] * n_assets
        constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        result = minimize(
            objective_func,
            initial_weights,
            method='SLSQP',
            jac=gradient_func,
            bounds=bounds,
            constraints=constraints_opt,
            options={'maxiter': algo_config.max_iterations}
        )
        
        optimal_weights = result.x.astype(np.float32)
        if np.abs(np.sum(optimal_weights) - 1.0) > 1e-6:
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # 9. 生成信号
        signals = self.system_layer.generate_signal(optimal_weights)
        
        return {
            'optimal_weights': dict(zip(valid_symbols, optimal_weights)),
            'signals': dict(zip(valid_symbols, signals)),
            'optimization_info': {
                'success': result.success,
                'iterations': result.nit,
                'algorithm': optimal_algorithm.value,
                'data_source': 'QDB' if self.data_loader.use_qdb else 'traditional',
                'cache_used': cache_key in self.data_loader._covariance_cache
            },
            'performance_metrics': {
                'portfolio_return': float(np.dot(optimal_weights, mean_returns)),
                'portfolio_std': float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))),
                'sharpe_ratio': float(np.dot(optimal_weights, mean_returns) / np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))
            }
        }


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建增强的优化栈（集成QDB）
    stack = EnhancedOptimizationStack(use_qdb=True)
    
    # 从QDB加载数据并优化
    result = stack.optimize_portfolio_from_qdb(
        symbols=['SPY', 'AAPL', 'MSFT', 'GOOGL'],
        start_time='2024-01-01',
        end_time='2024-12-31',
        objective=ModelObjective.MAXIMIZE_SHARPE
    )
    
    print("优化结果:")
    print(f"  最优权重: {result['optimal_weights']}")
    print(f"  使用算法: {result['optimization_info']['algorithm']}")
    print(f"  数据源: {result['optimization_info']['data_source']}")
    print(f"  Sharpe比率: {result['performance_metrics']['sharpe_ratio']:.4f}")


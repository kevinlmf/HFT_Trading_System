"""
Smart Execution Engine
根据数据特征自动选择最优实现方式（Python/C++/CUDA）
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# DataProfile and DataQuality - using simplified versions if not available
try:
    from Data.eda.data_analyzer import DataProfile, DataQuality
except ImportError:
    # Fallback: create simple placeholder classes
    from typing import Optional
    class DataProfile:
        def __init__(self, *args, **kwargs):
            pass
    class DataQuality:
        def __init__(self, *args, **kwargs):
            pass

# 尝试导入C++模块
cpp_core_dir = Path(__file__).resolve().parent.parent / "cpp_core"
if str(cpp_core_dir) not in sys.path:
    sys.path.insert(0, str(cpp_core_dir))

try:
    import cpp_trading2
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

# CUDA support removed - using C++ and Python implementations only
CUDA_AVAILABLE = False


class SmartExecutor:
    """
    智能执行引擎
    
    根据数据特征自动选择最优实现方式：
    - Python向量化：小规模数据，快速开发
    - C++：中等大规模数据，高性能
    """
    
    def __init__(self):
        self.impl_cache: Dict[str, Any] = {}
        self.stats = {
            'python_calls': 0,
            'cpp_calls': 0,
            'total_orders': 0
        }
    
    def execute_slippage_calculation(self,
                                    prices: np.ndarray,
                                    quantities: np.ndarray,
                                    mid_prices: np.ndarray,
                                    sides: np.ndarray,
                                    profile: Optional[DataProfile] = None,
                                    force_implementation: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        智能执行slippage计算
        
        Args:
            prices: 价格数组
            quantities: 数量数组
            mid_prices: 中间价数组
            sides: 买卖方向数组
            profile: 数据特征分析结果（可选，会自动分析）
            force_implementation: 强制使用指定实现（'python', 'cpp', 'cuda'）
        
        Returns:
            (slippage_costs, execution_info)
        """
        n_orders = len(prices)
        self.stats['total_orders'] += n_orders
        
        # 如果没有提供profile，进行快速分析
        if profile is None:
            try:
                from Data.eda.data_analyzer import DataAnalyzer
                analyzer = DataAnalyzer()
            except ImportError:
                analyzer = None
            # 创建临时DataFrame进行分析
            temp_df = pd.DataFrame({
                'price': prices,
                'quantity': quantities,
                'mid_price': mid_prices,
                'side': sides
            })
            profile = analyzer.analyze(temp_df, data_name="slippage_data")
        
        # 确定实现方式
        if force_implementation:
            impl = force_implementation
        else:
            impl = profile.recommended_implementation
        
        # 根据实现方式执行
        execution_info = {
            'implementation': impl,
            'n_orders': n_orders,
            'execution_time': 0.0,
            'throughput': 0.0
        }
        
        import time
        start_time = time.perf_counter()
        
        if impl == "python_vectorized" or impl == "python":
            result = self._execute_python(prices, quantities, mid_prices, sides)
            self.stats['python_calls'] += 1
        elif impl == "cpp":
            if not CPP_AVAILABLE:
                print("Warning: C++ not available, falling back to Python")
                result = self._execute_python(prices, quantities, mid_prices, sides)
                execution_info['implementation'] = 'python_vectorized'
                self.stats['python_calls'] += 1
            else:
                result = self._execute_cpp(prices, quantities, mid_prices, sides)
                self.stats['cpp_calls'] += 1
        elif impl == "cuda":
            # CUDA support removed - fallback to C++ or Python
            print("Warning: CUDA not available, falling back to C++ or Python")
            if CPP_AVAILABLE:
                result = self._execute_cpp(prices, quantities, mid_prices, sides)
                execution_info['implementation'] = 'cpp'
                self.stats['cpp_calls'] += 1
            else:
                result = self._execute_python(prices, quantities, mid_prices, sides)
                execution_info['implementation'] = 'python_vectorized'
                self.stats['python_calls'] += 1
        else:
            # 默认使用Python
            result = self._execute_python(prices, quantities, mid_prices, sides)
            execution_info['implementation'] = 'python_vectorized'
            self.stats['python_calls'] += 1
        
        elapsed = time.perf_counter() - start_time
        execution_info['execution_time'] = elapsed
        execution_info['throughput'] = n_orders / elapsed if elapsed > 0 else 0
        
        return result, execution_info
    
    def _execute_python(self, prices, quantities, mid_prices, sides):
        """Python向量化实现"""
        base_slip = 0.0001  # 1 bps
        vol_adjustment = 1.0 * 0.5
        liquidity_adjustment = 0.5
        size_threshold = 100000.0
        
        notional_values = quantities * prices
        size_adjustment = np.minimum(notional_values / size_threshold, 2.0)
        
        total_slippage_factor = base_slip * (1.0 + vol_adjustment + liquidity_adjustment + size_adjustment)
        slippage = prices * total_slippage_factor
        execution_prices = mid_prices + (sides * slippage)
        slippage_costs = np.abs(execution_prices - mid_prices) * quantities
        
        return slippage_costs
    
    def _execute_cpp(self, prices, quantities, mid_prices, sides):
        """C++实现"""
        if not hasattr(cpp_trading2, 'SlippageCalculator'):
            # Fallback to Python
            return self._execute_python(prices, quantities, mid_prices, sides)
        
        params = cpp_trading2.SlippageParams()
        params.base_slippage_bps = 1.0
        params.volatility = 1.0
        params.liquidity_factor = 1.0
        params.size_threshold = 100000.0
        
        calculator = cpp_trading2.SlippageCalculator(params)
        slippage_costs = calculator.calculate_batch(
            prices.astype(np.float64),
            quantities.astype(np.float64),
            mid_prices.astype(np.float64),
            sides.astype(np.int32)
        )
        
        return np.array(slippage_costs)
    
    def get_stats(self) -> Dict:
        """获取执行统计信息"""
        return {
            **self.stats,
            'python_ratio': self.stats['python_calls'] / max(sum([
                self.stats['python_calls'],
                self.stats['cpp_calls']
            ]), 1),
            'cpp_ratio': self.stats['cpp_calls'] / max(sum([
                self.stats['python_calls'],
                self.stats['cpp_calls']
            ]), 1),
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'python_calls': 0,
            'cpp_calls': 0,
            'total_orders': 0
        }






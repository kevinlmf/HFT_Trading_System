"""
CUDA Slippage计算Python包装
提供Python接口调用CUDA kernel进行批量slippage计算
"""
import numpy as np
import ctypes
from pathlib import Path
import os

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Install with: pip install cupy")

# 尝试加载CUDA库
_cuda_lib = None
_cuda_lib_path = Path(__file__).parent.parent / "build" / "libslippage.so"

if _cuda_lib_path.exists():
    try:
        _cuda_lib = ctypes.CDLL(str(_cuda_lib_path))
        _cuda_lib.calculate_slippage_batch_cuda.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # prices
            ctypes.POINTER(ctypes.c_float),  # quantities
            ctypes.POINTER(ctypes.c_float),  # mid_prices
            ctypes.POINTER(ctypes.c_float),  # sides
            ctypes.POINTER(ctypes.c_float),  # slippage_costs (output)
            ctypes.c_int,                     # n_orders
            ctypes.c_float,                   # base_slippage_bps
            ctypes.c_float,                   # volatility
            ctypes.c_float,                   # liquidity_factor
            ctypes.c_float,                   # size_threshold
            ctypes.c_bool                     # use_optimized
        ]
        _cuda_lib.calculate_slippage_batch_cuda.restype = None
        CUDA_LIB_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not load CUDA library: {e}")
        CUDA_LIB_AVAILABLE = False
else:
    CUDA_LIB_AVAILABLE = False


class CUDASlippageCalculator:
    """CUDA加速的Slippage计算器"""
    
    def __init__(self,
                 base_slippage_bps: float = 1.0,
                 volatility: float = 1.0,
                 liquidity_factor: float = 1.0,
                 size_threshold: float = 100000.0,
                 use_cupy: bool = True):
        """
        初始化CUDA Slippage计算器
        
        Args:
            base_slippage_bps: 基础slippage (basis points)
            volatility: 波动率调整因子
            liquidity_factor: 流动性因子
            size_threshold: 订单规模阈值
            use_cupy: 是否使用CuPy（如果可用）
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility = volatility
        self.liquidity_factor = liquidity_factor
        self.size_threshold = size_threshold
        self.use_cupy = use_cupy and CUPY_AVAILABLE
        
        if self.use_cupy:
            # 使用CuPy实现（更简单，但需要CuPy）
            pass
        elif CUDA_LIB_AVAILABLE:
            # 使用编译的CUDA库
            pass
        else:
            print("Warning: CUDA not available. Falling back to CPU.")
    
    def calculate_batch(self,
                       prices: np.ndarray,
                       quantities: np.ndarray,
                       mid_prices: np.ndarray,
                       sides: np.ndarray) -> np.ndarray:
        """
        批量计算slippage
        
        Args:
            prices: 价格数组
            quantities: 数量数组
            mid_prices: 中间价数组
            sides: 买卖方向数组 (1.0=buy, -1.0=sell)
        
        Returns:
            slippage_costs: slippage成本数组
        """
        n = len(prices)
        
        if self.use_cupy:
            return self._calculate_with_cupy(prices, quantities, mid_prices, sides)
        elif CUDA_LIB_AVAILABLE:
            return self._calculate_with_lib(prices, quantities, mid_prices, sides)
        else:
            # Fallback to numpy (CPU)
            return self._calculate_cpu(prices, quantities, mid_prices, sides)
    
    def _calculate_with_cupy(self,
                            prices: np.ndarray,
                            quantities: np.ndarray,
                            mid_prices: np.ndarray,
                            sides: np.ndarray) -> np.ndarray:
        """使用CuPy在GPU上计算"""
        # 转换为float32并移到GPU
        prices_gpu = cp.asarray(prices.astype(np.float32))
        quantities_gpu = cp.asarray(quantities.astype(np.float32))
        mid_prices_gpu = cp.asarray(mid_prices.astype(np.float32))
        sides_gpu = cp.asarray(sides.astype(np.float32))
        
        # 计算参数
        base_slip = self.base_slippage_bps / 10000.0
        vol_adjustment = self.volatility * 0.5
        liquidity_adjustment = (2.0 - self.liquidity_factor) * 0.5
        
        # 批量计算
        notional_values = quantities_gpu * prices_gpu
        size_adjustment = cp.minimum(notional_values / self.size_threshold, 2.0)
        
        total_slippage_factor = base_slip * (1.0 + vol_adjustment + liquidity_adjustment + size_adjustment)
        slippage = prices_gpu * total_slippage_factor
        execution_prices = mid_prices_gpu + (sides_gpu * slippage)
        slippage_costs = cp.abs(execution_prices - mid_prices_gpu) * quantities_gpu
        
        # 移回CPU
        return cp.asnumpy(slippage_costs).astype(np.float64)
    
    def _calculate_with_lib(self,
                           prices: np.ndarray,
                           quantities: np.ndarray,
                           mid_prices: np.ndarray,
                           sides: np.ndarray) -> np.ndarray:
        """使用编译的CUDA库计算"""
        n = len(prices)
        
        # 转换为float32
        prices_f32 = prices.astype(np.float32)
        quantities_f32 = quantities.astype(np.float32)
        mid_prices_f32 = mid_prices.astype(np.float32)
        sides_f32 = sides.astype(np.float32)
        
        # 分配GPU内存
        prices_gpu = cuda.mem_alloc(prices_f32.nbytes)
        quantities_gpu = cuda.mem_alloc(quantities_f32.nbytes)
        mid_prices_gpu = cuda.mem_alloc(mid_prices_f32.nbytes)
        sides_gpu = cuda.mem_alloc(sides_f32.nbytes)
        slippage_costs_gpu = cuda.mem_alloc(n * 4)  # float32 = 4 bytes
        
        # 复制数据到GPU
        cuda.memcpy_htod(prices_gpu, prices_f32)
        cuda.memcpy_htod(quantities_gpu, quantities_f32)
        cuda.memcpy_htod(mid_prices_gpu, mid_prices_f32)
        cuda.memcpy_htod(sides_gpu, sides_f32)
        
        # 调用kernel
        _cuda_lib.calculate_slippage_batch_cuda(
            ctypes.cast(int(prices_gpu), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(int(quantities_gpu), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(int(mid_prices_gpu), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(int(sides_gpu), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(int(slippage_costs_gpu), ctypes.POINTER(ctypes.c_float)),
            n,
            ctypes.c_float(self.base_slippage_bps),
            ctypes.c_float(self.volatility),
            ctypes.c_float(self.liquidity_factor),
            ctypes.c_float(self.size_threshold),
            ctypes.c_bool(True)
        )
        
        # 复制结果回CPU
        slippage_costs = np.empty(n, dtype=np.float32)
        cuda.memcpy_dtoh(slippage_costs, slippage_costs_gpu)
        
        # 释放GPU内存
        prices_gpu.free()
        quantities_gpu.free()
        mid_prices_gpu.free()
        sides_gpu.free()
        slippage_costs_gpu.free()
        
        return slippage_costs.astype(np.float64)
    
    def _calculate_cpu(self,
                      prices: np.ndarray,
                      quantities: np.ndarray,
                      mid_prices: np.ndarray,
                      sides: np.ndarray) -> np.ndarray:
        """CPU fallback（使用numpy向量化）"""
        base_slip = self.base_slippage_bps / 10000.0
        vol_adjustment = self.volatility * 0.5
        liquidity_adjustment = (2.0 - self.liquidity_factor) * 0.5
        
        notional_values = quantities * prices
        size_adjustment = np.minimum(notional_values / self.size_threshold, 2.0)
        
        total_slippage_factor = base_slip * (1.0 + vol_adjustment + liquidity_adjustment + size_adjustment)
        slippage = prices * total_slippage_factor
        execution_prices = mid_prices + (sides * slippage)
        slippage_costs = np.abs(execution_prices - mid_prices) * quantities
        
        return slippage_costs


# 便捷函数
def calculate_slippage_cuda(prices: np.ndarray,
                           quantities: np.ndarray,
                           mid_prices: np.ndarray,
                           sides: np.ndarray,
                           base_slippage_bps: float = 1.0,
                           volatility: float = 1.0,
                           liquidity_factor: float = 1.0) -> np.ndarray:
    """
    便捷函数：使用CUDA计算slippage
    
    Args:
        prices: 价格数组
        quantities: 数量数组
        mid_prices: 中间价数组
        sides: 买卖方向数组 (1.0=buy, -1.0=sell)
        base_slippage_bps: 基础slippage (basis points)
        volatility: 波动率调整因子
        liquidity_factor: 流动性因子
    
    Returns:
        slippage_costs: slippage成本数组
    """
    calculator = CUDASlippageCalculator(
        base_slippage_bps=base_slippage_bps,
        volatility=volatility,
        liquidity_factor=liquidity_factor
    )
    return calculator.calculate_batch(prices, quantities, mid_prices, sides)








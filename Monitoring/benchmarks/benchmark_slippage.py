"""
Slippage Performance Benchmark

Compare Python loop, NumPy vectorised, C++, and CUDA implementations of the
slippage calculation. Lower latency is critical for HFT execution, so we
capture execution time, throughput, and speed-up across multiple batch sizes.
"""
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Ensure in-place C++ builds can be imported without pip install
cpp_core_dir = Path(__file__).parent.parent.parent / "Execution" / "cpp_core"
if cpp_core_dir.exists():
    if str(cpp_core_dir) not in sys.path:
        sys.path.insert(0, str(cpp_core_dir))

# 尝试导入C++模块
try:
    import cpp_trading2
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ module not available. Install with: cd Execution/cpp_core && pip install -e .")

# 尝试导入CUDA模块
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cuda_accelerated" / "python"))
    from cuda_slippage import CUDASlippageCalculator, calculate_slippage_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA module not available. Ensure CUDA is properly installed.")

# 导入Python实现
from Environment.simulator.market_simulator import RealisticMarketSimulator, OrderSide, OrderType


class SlippageBenchmark:
    """Slippage计算性能基准测试"""
    
    def __init__(self):
        self.results = {}
        
    def generate_test_data(self, n_orders: int, n_symbols: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成测试数据
        
        Returns:
            prices: 价格数组 [n_orders]
            quantities: 数量数组 [n_orders]
            mid_prices: 中间价数组 [n_orders]
            sides: 买卖方向数组 [n_orders] (1=buy, -1=sell)
        """
        np.random.seed(42)
        
        # 生成价格数据（模拟股票价格）
        base_prices = np.random.uniform(50, 200, n_symbols)
        prices = np.random.choice(base_prices, n_orders)
        prices += np.random.randn(n_orders) * 2  # 添加价格波动
        
        # 生成数量
        quantities = np.random.uniform(10, 1000, n_orders)
        
        # 生成中间价（价格附近）
        mid_prices = prices + np.random.randn(n_orders) * 0.5
        
        # 生成买卖方向
        sides = np.random.choice([1, -1], n_orders)
        
        return prices, quantities, mid_prices, sides
    
    def calculate_slippage_python(self, prices: np.ndarray, quantities: np.ndarray, 
                                   mid_prices: np.ndarray, sides: np.ndarray,
                                   base_slippage_bps: float = 1.0,
                                   volatility: float = 1.0,
                                   liquidity_factor: float = 1.0) -> np.ndarray:
        """
        Python版本的slippage计算（参考market_simulator.py的实现）
        """
        slippage_factors = np.zeros(len(prices))
        
        for i in range(len(prices)):
            # Base slippage
            base_slip = base_slippage_bps / 10000.0  # 转换为小数
            
            # Volatility adjustment
            vol_adjustment = volatility * 0.5
            
            # Liquidity adjustment
            liquidity_adjustment = (2.0 - liquidity_factor) * 0.5
            
            # Order size adjustment
            notional_value = quantities[i] * prices[i]
            size_adjustment = min(notional_value / 100000.0, 2.0)
            
            # Total slippage factor
            total_slippage_factor = base_slip * (1 + vol_adjustment + liquidity_adjustment + size_adjustment)
            
            # Calculate actual slippage
            slippage = prices[i] * total_slippage_factor
            
            # Apply direction
            if sides[i] == 1:  # Buy
                execution_price = mid_prices[i] + slippage
            else:  # Sell
                execution_price = mid_prices[i] - slippage
            
            slippage_factors[i] = abs(execution_price - mid_prices[i]) * quantities[i]
        
        return slippage_factors
    
    def calculate_slippage_python_vectorized(self, prices: np.ndarray, quantities: np.ndarray,
                                            mid_prices: np.ndarray, sides: np.ndarray,
                                            base_slippage_bps: float = 1.0,
                                            volatility: float = 1.0,
                                            liquidity_factor: float = 1.0) -> np.ndarray:
        """
        Python向量化版本的slippage计算（使用numpy）
        """
        # Base slippage
        base_slip = base_slippage_bps / 10000.0
        
        # Volatility adjustment
        vol_adjustment = volatility * 0.5
        
        # Liquidity adjustment
        liquidity_adjustment = (2.0 - liquidity_factor) * 0.5
        
        # Order size adjustment
        notional_values = quantities * prices
        size_adjustment = np.minimum(notional_values / 100000.0, 2.0)
        
        # Total slippage factor
        total_slippage_factor = base_slip * (1 + vol_adjustment + liquidity_adjustment + size_adjustment)
        
        # Calculate slippage
        slippage = prices * total_slippage_factor
        
        # Apply direction
        execution_prices = mid_prices + (sides * slippage)
        
        # Calculate total slippage cost
        slippage_costs = np.abs(execution_prices - mid_prices) * quantities
        
        return slippage_costs
    
    def calculate_slippage_cpp(self, prices: np.ndarray, quantities: np.ndarray,
                               mid_prices: np.ndarray, sides: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        C++版本的slippage计算（使用SlippageCalculator批量计算）
        """
        if not CPP_AVAILABLE:
            return np.array([]), 0.0
        
        # 检查是否有SlippageCalculator
        if not hasattr(cpp_trading2, 'SlippageCalculator'):
            # Fallback to old method
            return self._calculate_slippage_cpp_old(prices, quantities, mid_prices, sides)
        
        start_time = time.perf_counter()
        
        # 创建slippage计算器
        params = cpp_trading2.SlippageParams()
        params.base_slippage_bps = 1.0
        params.volatility = 1.0
        params.liquidity_factor = 1.0
        params.size_threshold = 100000.0
        
        calculator = cpp_trading2.SlippageCalculator(params)
        
        # 批量计算
        slippage_costs = calculator.calculate_batch(
            prices.astype(np.float64),
            quantities.astype(np.float64),
            mid_prices.astype(np.float64),
            sides.astype(np.int32)
        )
        
        elapsed = time.perf_counter() - start_time
        
        return np.array(slippage_costs), elapsed
    
    def _calculate_slippage_cpp_old(self, prices: np.ndarray, quantities: np.ndarray,
                                    mid_prices: np.ndarray, sides: np.ndarray) -> Tuple[np.ndarray, float]:
        """旧的C++实现（使用order_executor）"""
        start_time = time.perf_counter()
        
        # 创建order executor
        executor = cpp_trading2.OrderExecutor(commission_rate=0.0001, min_commission=0.0)
        
        # 创建orderbook
        orderbook = cpp_trading2.FastOrderBook()
        
        slippage_costs = []
        
        for i in range(len(prices)):
            # 设置orderbook
            orderbook.reset()
            orderbook.add_bid(mid_prices[i] - 0.01, 10000)  # 模拟bid
            orderbook.add_ask(mid_prices[i] + 0.01, 10000)  # 模拟ask
            
            # 提交订单
            side = cpp_trading2.OrderSide.BUY if sides[i] > 0 else cpp_trading2.OrderSide.SELL
            order_id = executor.submit_order(side, cpp_trading2.HFTOrderType.MARKET, quantities[i])
            
            # 执行订单
            fills = executor.execute_pending_orders(orderbook)
            
            if fills:
                fill = fills[0]
                slippage_costs.append(fill.slippage * fill.executed_qty)
            else:
                slippage_costs.append(0.0)
        
        elapsed = time.perf_counter() - start_time
        
        return np.array(slippage_costs), elapsed
    
    def calculate_slippage_cuda(self, prices: np.ndarray, quantities: np.ndarray,
                                mid_prices: np.ndarray, sides: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        CUDA版本的slippage计算（使用CUDA加速）
        """
        if not CUDA_AVAILABLE:
            return np.array([]), 0.0
        
        start_time = time.perf_counter()
        
        # 使用CUDA计算器
        calculator = CUDASlippageCalculator(
            base_slippage_bps=1.0,
            volatility=1.0,
            liquidity_factor=1.0,
            size_threshold=100000.0
        )
        
        slippage_costs = calculator.calculate_batch(prices, quantities, mid_prices, sides)
        
        elapsed = time.perf_counter() - start_time
        
        return slippage_costs, elapsed
    
    def benchmark(self, n_orders_list: List[int], n_runs: int = 5) -> Dict:
        """
        运行性能基准测试
        
        Args:
            n_orders_list: 不同订单数量的列表
            n_runs: 每个测试运行的次数（取平均值）
        """
        results = {
            'n_orders': [],
            'python_loop': {'times': [], 'throughput': []},
            'python_vectorized': {'times': [], 'throughput': []},
            'cpp': {'times': [], 'throughput': []},
            'cuda': {'times': [], 'throughput': []}
        }
        
        for n_orders in n_orders_list:
            print(f"\nOrders per batch: {n_orders:,}")
            results['n_orders'].append(n_orders)
            
            # 生成测试数据
            prices, quantities, mid_prices, sides = self.generate_test_data(n_orders)
            
            # Python循环版本
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.calculate_slippage_python(prices, quantities, mid_prices, sides)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            results['python_loop']['times'].append(avg_time)
            results['python_loop']['throughput'].append(n_orders / avg_time)
            print(f"  Python (loop):  {avg_time*1000:.2f} ms, {n_orders/avg_time:.0f} orders/sec")
            
            # Python向量化版本
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.calculate_slippage_python_vectorized(prices, quantities, mid_prices, sides)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_time = np.mean(times)
            results['python_vectorized']['times'].append(avg_time)
            results['python_vectorized']['throughput'].append(n_orders / avg_time)
            print(f"  Python (NumPy): {avg_time*1000:.2f} ms, {n_orders/avg_time:.0f} orders/sec")
            
            # C++版本
            if CPP_AVAILABLE:
                times = []
                for _ in range(n_runs):
                    _, elapsed = self.calculate_slippage_cpp(prices, quantities, mid_prices, sides)
                    times.append(elapsed)
                avg_time = np.mean(times)
                results['cpp']['times'].append(avg_time)
                results['cpp']['throughput'].append(n_orders / avg_time)
                print(f"  C++:            {avg_time*1000:.2f} ms, {n_orders/avg_time:.0f} orders/sec")
            else:
                results['cpp']['times'].append(np.nan)
                results['cpp']['throughput'].append(np.nan)
            
            # CUDA版本
            if CUDA_AVAILABLE:
                times = []
                for _ in range(n_runs):
                    _, elapsed = self.calculate_slippage_cuda(prices, quantities, mid_prices, sides)
                    times.append(elapsed)
                avg_time = np.mean(times)
                results['cuda']['times'].append(avg_time)
                results['cuda']['throughput'].append(n_orders / avg_time)
                print(f"  CUDA:           {avg_time*1000:.2f} ms, {n_orders/avg_time:.0f} orders/sec")
            else:
                results['cuda']['times'].append(np.nan)
                results['cuda']['throughput'].append(np.nan)
        
        self.results = results
        return results
    
    def calculate_speedup(self) -> Dict:
        """计算加速比"""
        if not self.results:
            return {}
        
        speedup = {}
        n_orders = self.results['n_orders']
        
        python_base_times = np.array(self.results['python_loop']['times'])
        
        for impl in ['python_vectorized', 'cpp', 'cuda']:
            if impl in self.results:
                times = np.array(self.results[impl]['times'])
                valid_mask = ~np.isnan(times)
                if np.any(valid_mask):
                    speedup[impl] = {
                        'n_orders': [n_orders[i] for i in range(len(n_orders)) if valid_mask[i]],
                        'speedup': (python_base_times[valid_mask] / times[valid_mask]).tolist()
                    }
        
        return speedup
    
    def plot_results(self, save_path: str = None):
        """Plot benchmark results."""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Slippage Performance Benchmark', fontsize=16, fontweight='bold')
        
        n_orders = self.results['n_orders']
        
        # 1. 执行时间对比
        ax1 = axes[0, 0]
        ax1.plot(n_orders, np.array(self.results['python_loop']['times']) * 1000,
                'o-', label='Python (loop)', linewidth=2, markersize=8)
        ax1.plot(n_orders, np.array(self.results['python_vectorized']['times']) * 1000,
                's-', label='Python (NumPy)', linewidth=2, markersize=8)
        
        if CPP_AVAILABLE and not np.isnan(self.results['cpp']['times'][0]):
            ax1.plot(n_orders, np.array(self.results['cpp']['times']) * 1000,
                    '^-', label='C++', linewidth=2, markersize=8)
        
        if CUDA_AVAILABLE and not np.isnan(self.results['cuda']['times'][0]):
            ax1.plot(n_orders, np.array(self.results['cuda']['times']) * 1000,
                    'd-', label='CUDA', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Order count', fontsize=12)
        ax1.set_ylabel('Execution time (ms)', fontsize=12)
        ax1.set_title('Execution time', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 2. 吞吐量对比
        ax2 = axes[0, 1]
        ax2.plot(n_orders, self.results['python_loop']['throughput'],
                'o-', label='Python (loop)', linewidth=2, markersize=8)
        ax2.plot(n_orders, self.results['python_vectorized']['throughput'],
                's-', label='Python (NumPy)', linewidth=2, markersize=8)
        
        if CPP_AVAILABLE and not np.isnan(self.results['cpp']['throughput'][0]):
            ax2.plot(n_orders, self.results['cpp']['throughput'],
                    '^-', label='C++', linewidth=2, markersize=8)
        
        if CUDA_AVAILABLE and not np.isnan(self.results['cuda']['throughput'][0]):
            ax2.plot(n_orders, self.results['cuda']['throughput'],
                    'd-', label='CUDA', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Order count', fontsize=12)
        ax2.set_ylabel('Throughput (orders/sec)', fontsize=12)
        ax2.set_title('Throughput', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # 3. Speed-up
        speedup = self.calculate_speedup()
        ax3 = axes[1, 0]
        
        if 'python_vectorized' in speedup:
            ax3.plot(speedup['python_vectorized']['n_orders'], 
                    speedup['python_vectorized']['speedup'],
                    's-', label='Python (NumPy) vs loop', linewidth=2, markersize=8)
        
        if 'cpp' in speedup:
            ax3.plot(speedup['cpp']['n_orders'], speedup['cpp']['speedup'],
                    '^-', label='C++ vs loop', linewidth=2, markersize=8)
        
        if 'cuda' in speedup:
            ax3.plot(speedup['cuda']['n_orders'], speedup['cuda']['speedup'],
                    'd-', label='CUDA vs loop', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Order count', fontsize=12)
        ax3.set_ylabel('Speed-up (x)', fontsize=12)
        ax3.set_title('Speed-up vs Python loop', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # 4. Peak throughput (largest batch)
        ax4 = axes[1, 1]
        
        python_throughput = self.results['python_loop']['throughput'][-1]
        python_vec_throughput = self.results['python_vectorized']['throughput'][-1]
        
        throughputs = [python_throughput, python_vec_throughput]
        labels = ['Python (loop)', 'Python (NumPy)']
        
        if CPP_AVAILABLE and not np.isnan(self.results['cpp']['throughput'][-1]):
            throughputs.append(self.results['cpp']['throughput'][-1])
            labels.append('C++')
        
        if CUDA_AVAILABLE and not np.isnan(self.results['cuda']['throughput'][-1]):
            throughputs.append(self.results['cuda']['throughput'][-1])
            labels.append('CUDA')
        
        bars = ax4.bar(labels, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax4.set_ylabel('Throughput (orders/sec)', fontsize=12)
        ax4.set_title('Peak throughput (largest batch)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Annotate bars with raw throughput values
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlots saved to: {save_path}")
        else:
            plt.show()
    
    def print_summary(self):
        """Print a summary of the latest benchmark run."""
        if not self.results:
            print("No results to summarize. Run benchmark first.")
            return
        
        print("\n" + "="*80)
        print("Slippage Benchmark Summary")
        print("="*80)
        
        n_orders = self.results['n_orders']
        max_idx = len(n_orders) - 1
        
        print(f"\nLargest batch size: {n_orders[max_idx]:,} orders")
        print("\nExecution time (ms):")
        print(f"  Python (loop):  {self.results['python_loop']['times'][max_idx]*1000:.2f} ms")
        print(f"  Python (NumPy): {self.results['python_vectorized']['times'][max_idx]*1000:.2f} ms")
        
        if CPP_AVAILABLE and not np.isnan(self.results['cpp']['times'][max_idx]):
            print(f"  C++:            {self.results['cpp']['times'][max_idx]*1000:.2f} ms")
        
        if CUDA_AVAILABLE and not np.isnan(self.results['cuda']['times'][max_idx]):
            print(f"  CUDA:           {self.results['cuda']['times'][max_idx]*1000:.2f} ms")
        
        print("\nThroughput (orders/sec):")
        print(f"  Python (loop):  {self.results['python_loop']['throughput'][max_idx]:,.0f}")
        print(f"  Python (NumPy): {self.results['python_vectorized']['throughput'][max_idx]:,.0f}")
        
        if CPP_AVAILABLE and not np.isnan(self.results['cpp']['throughput'][max_idx]):
            print(f"  C++:            {self.results['cpp']['throughput'][max_idx]:,.0f}")
        
        if CUDA_AVAILABLE and not np.isnan(self.results['cuda']['throughput'][max_idx]):
            print(f"  CUDA:           {self.results['cuda']['throughput'][max_idx]:,.0f}")
        
        # Speed-up ratios
        speedup = self.calculate_speedup()
        print("\nSpeed-up vs Python loop:")
        
        python_vec_speedup = speedup.get('python_vectorized', {}).get('speedup', [])
        if python_vec_speedup:
            print(f"  Python (NumPy): {python_vec_speedup[-1]:.2f}x")
        
        cpp_speedup = speedup.get('cpp', {}).get('speedup', [])
        if cpp_speedup:
            print(f"  C++:            {cpp_speedup[-1]:.2f}x")
        
        cuda_speedup = speedup.get('cuda', {}).get('speedup', [])
        if cuda_speedup:
            print(f"  CUDA:           {cuda_speedup[-1]:.2f}x")
        
        # HFT throughput target
        print("\n" + "-"*80)
        print("HFT throughput target (1,000 orders/sec)")
        print("-"*80)
        
        target_throughput = 1000
        python_time_per_order = 1.0 / self.results['python_loop']['throughput'][max_idx]
        python_vec_time_per_order = 1.0 / self.results['python_vectorized']['throughput'][max_idx]
        
        print(f"\nTarget throughput: {target_throughput:,} orders/sec")
        print(f"  Python (loop):  {python_time_per_order*1000:.3f} ms/order ({target_throughput*python_time_per_order*1000:.1f} ms per 1k orders)")
        print(f"  Python (NumPy): {python_vec_time_per_order*1000:.3f} ms/order ({target_throughput*python_vec_time_per_order*1000:.1f} ms per 1k orders)")
        
        if CPP_AVAILABLE and not np.isnan(self.results['cpp']['throughput'][max_idx]):
            cpp_time_per_order = 1.0 / self.results['cpp']['throughput'][max_idx]
            print(f"  C++:            {cpp_time_per_order*1000:.3f} ms/order ({target_throughput*cpp_time_per_order*1000:.1f} ms per 1k orders)")
        
        if CUDA_AVAILABLE and not np.isnan(self.results['cuda']['throughput'][max_idx]):
            cuda_time_per_order = 1.0 / self.results['cuda']['throughput'][max_idx]
            print(f"  CUDA:           {cuda_time_per_order*1000:.3f} ms/order ({target_throughput*cuda_time_per_order*1000:.1f} ms per 1k orders)")
        
        print("\n" + "="*80)


def main():
    """CLI entrypoint for the benchmark."""
    print("="*80)
    print("Slippage Performance Benchmark")
    print("="*80)
    print("\nCompare Python loop, Python (NumPy), C++, and CUDA implementations of the slippage calculation.")
    print("Lower latency is critical for HFT execution, so we measure each backend across multiple batch sizes.\n")
    
    benchmark = SlippageBenchmark()
    
    # Order counts to test
    n_orders_list = [100, 1000, 10000, 100000]
    
    print(f"Batch sizes: {n_orders_list}")
    print("Each measurement is averaged over 5 runs.\n")
    
    # Execute the benchmark
    benchmark.benchmark(n_orders_list, n_runs=5)
    
    # Print summary
    benchmark.print_summary()
    
    # Plot findings
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "slippage_benchmark_results.png"
    benchmark.plot_results(save_path=str(plot_path))
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()


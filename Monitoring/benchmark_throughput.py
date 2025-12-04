"""
High-Throughput Data Processing Benchmark
测试系统处理大量市场数据的能力
"""
import sys
import os
# Add parent directory to path for cpp_trading2 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import cpp_trading2
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    name: str
    data_points: int
    elapsed_time: float
    throughput: float  # operations per second
    latency_mean: float  # microseconds
    latency_p50: float
    latency_p95: float
    latency_p99: float


class ThroughputBenchmark:
    """Comprehensive throughput benchmarking suite"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark_orderbook_updates(self, n_updates: int = 1_000_000) -> BenchmarkResult:
        """
        测试订单簿更新速度
        模拟高频市场数据流
        """
        print(f"\n📊 Benchmarking OrderBook Updates ({n_updates:,} updates)...")

        orderbook = cpp_trading2.FastOrderBook()

        # 生成测试数据
        np.random.seed(42)
        sides = np.random.choice([0, 1], n_updates)  # 0=buy, 1=sell
        prices = np.random.uniform(99.0, 101.0, n_updates)
        quantities = np.random.uniform(1.0, 100.0, n_updates)

        # 预热
        for i in range(100):
            if sides[i] == 0:
                orderbook.add_bid(prices[i], quantities[i])
            else:
                orderbook.add_ask(prices[i], quantities[i])

        # 测量批量更新
        latencies = []
        start = time.perf_counter()

        for i in range(n_updates):
            t0 = time.perf_counter()
            if sides[i] == 0:
                orderbook.add_bid(prices[i], quantities[i])
            else:
                orderbook.add_ask(prices[i], quantities[i])
            latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds

        elapsed = time.perf_counter() - start
        throughput = n_updates / elapsed

        latencies_arr = np.array(latencies)
        result = BenchmarkResult(
            name="OrderBook Updates",
            data_points=n_updates,
            elapsed_time=elapsed,
            throughput=throughput,
            latency_mean=latencies_arr.mean(),
            latency_p50=np.percentile(latencies_arr, 50),
            latency_p95=np.percentile(latencies_arr, 95),
            latency_p99=np.percentile(latencies_arr, 99)
        )

        self.results.append(result)
        return result

    def benchmark_order_execution(self, n_orders: int = 500_000) -> BenchmarkResult:
        """
        测试订单执行速度
        """
        print(f"\n⚡ Benchmarking Order Execution ({n_orders:,} orders)...")

        executor = cpp_trading2.OrderExecutor()
        orderbook = cpp_trading2.FastOrderBook()

        # 初始化订单簿
        for i in range(1000):
            orderbook.add_bid(100.0 - i * 0.01, 100.0)
            orderbook.add_ask(100.0 + i * 0.01, 100.0)

        # 生成测试订单
        np.random.seed(42)
        order_types = [cpp_trading2.HFTOrderType.MARKET] * (n_orders // 2) + \
                      [cpp_trading2.HFTOrderType.LIMIT] * (n_orders // 2)
        np.random.shuffle(order_types)

        sides = np.random.choice([cpp_trading2.OrderSide.BUY, cpp_trading2.OrderSide.SELL], n_orders)
        prices = np.random.uniform(99.0, 101.0, n_orders)
        quantities = np.random.uniform(1.0, 10.0, n_orders)

        # 预热
        for i in range(100):
            executor.submit_order(sides[i], order_types[i], quantities[i], prices[i])
            executor.execute_pending_orders(orderbook)

        # 重置执行器
        executor.reset()

        # 测量执行速度
        latencies = []
        start = time.perf_counter()

        for i in range(n_orders):
            t0 = time.perf_counter()
            executor.submit_order(sides[i], order_types[i], quantities[i], prices[i])
            executor.execute_pending_orders(orderbook)
            latencies.append((time.perf_counter() - t0) * 1e6)

        elapsed = time.perf_counter() - start
        throughput = n_orders / elapsed

        latencies_arr = np.array(latencies)
        result = BenchmarkResult(
            name="Order Execution",
            data_points=n_orders,
            elapsed_time=elapsed,
            throughput=throughput,
            latency_mean=latencies_arr.mean(),
            latency_p50=np.percentile(latencies_arr, 50),
            latency_p95=np.percentile(latencies_arr, 95),
            latency_p99=np.percentile(latencies_arr, 99)
        )

        self.results.append(result)
        return result

    def benchmark_jax_feature_computation(self, n_samples: int = 10_000_000) -> BenchmarkResult:
        """
        测试 JAX 特征计算速度
        模拟策略需要的大规模并行计算
        """
        print(f"\n🔢 Benchmarking JAX Feature Computation ({n_samples:,} samples)...")

        # 定义特征计算函数
        @jit
        def compute_features(prices, volumes, returns):
            """计算多个技术指标"""
            # 移动平均
            ma_5 = jnp.convolve(prices, jnp.ones(5)/5, mode='same')
            ma_20 = jnp.convolve(prices, jnp.ones(20)/20, mode='same')

            # RSI
            delta = jnp.diff(prices, prepend=prices[0])
            gain = jnp.where(delta > 0, delta, 0)
            loss = jnp.where(delta < 0, -delta, 0)
            avg_gain = jnp.convolve(gain, jnp.ones(14)/14, mode='same')
            avg_loss = jnp.convolve(loss, jnp.ones(14)/14, mode='same')
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # VWAP
            vwap = jnp.cumsum(prices * volumes) / jnp.cumsum(volumes)

            # 波动率
            volatility = jnp.std(returns)

            # 组合特征
            features = jnp.stack([ma_5, ma_20, rsi, vwap], axis=-1)
            return features, volatility

        # 生成测试数据
        key = jax.random.PRNGKey(42)
        prices = jax.random.uniform(key, (n_samples,), minval=99.0, maxval=101.0)
        volumes = jax.random.uniform(key, (n_samples,), minval=100.0, maxval=10000.0)
        returns = jax.random.normal(key, (n_samples,)) * 0.001

        # 预热 JIT
        _ = compute_features(prices[:1000], volumes[:1000], returns[:1000])
        jax.block_until_ready(_)

        # 批量处理测试
        batch_size = 100_000
        n_batches = n_samples // batch_size

        latencies = []
        start = time.perf_counter()

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            t0 = time.perf_counter()
            features, vol = compute_features(
                prices[start_idx:end_idx],
                volumes[start_idx:end_idx],
                returns[start_idx:end_idx]
            )
            jax.block_until_ready(features)
            latencies.append((time.perf_counter() - t0) * 1e6)

        elapsed = time.perf_counter() - start
        throughput = n_samples / elapsed

        latencies_arr = np.array(latencies)
        result = BenchmarkResult(
            name="JAX Feature Computation",
            data_points=n_samples,
            elapsed_time=elapsed,
            throughput=throughput,
            latency_mean=latencies_arr.mean(),
            latency_p50=np.percentile(latencies_arr, 50),
            latency_p95=np.percentile(latencies_arr, 95),
            latency_p99=np.percentile(latencies_arr, 99)
        )

        self.results.append(result)
        return result

    def benchmark_market_data_ingestion(self, n_ticks: int = 5_000_000) -> BenchmarkResult:
        """
        测试市场数据摄入速度
        模拟实时行情数据流处理
        """
        print(f"\n📈 Benchmarking Market Data Ingestion ({n_ticks:,} ticks)...")

        # 生成模拟 tick 数据（模拟从交易所接收数据）
        np.random.seed(42)
        timestamps = np.arange(n_ticks, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n_ticks) * 0.01)
        volumes = np.random.uniform(1.0, 1000.0, n_ticks)
        bids = prices - np.random.uniform(0.01, 0.05, n_ticks)
        asks = prices + np.random.uniform(0.01, 0.05, n_ticks)

        # 创建存储结构（模拟数据接收）
        market_data = []

        # 预热
        for i in range(100):
            tick = {
                'timestamp': timestamps[i],
                'price': prices[i],
                'volume': volumes[i],
                'bid': bids[i],
                'ask': asks[i]
            }
            market_data.append(tick)

        market_data.clear()

        # 测量数据摄入和处理速度
        latencies = []
        start = time.perf_counter()

        for i in range(n_ticks):
            t0 = time.perf_counter()

            # 模拟数据接收和基本处理
            tick = {
                'timestamp': timestamps[i],
                'price': prices[i],
                'volume': volumes[i],
                'bid': bids[i],
                'ask': asks[i],
                'mid': (bids[i] + asks[i]) / 2.0,
                'spread': asks[i] - bids[i]
            }
            market_data.append(tick)

            latencies.append((time.perf_counter() - t0) * 1e6)

        elapsed = time.perf_counter() - start
        throughput = n_ticks / elapsed

        latencies_arr = np.array(latencies)
        result = BenchmarkResult(
            name="Market Data Ingestion",
            data_points=n_ticks,
            elapsed_time=elapsed,
            throughput=throughput,
            latency_mean=latencies_arr.mean(),
            latency_p50=np.percentile(latencies_arr, 50),
            latency_p95=np.percentile(latencies_arr, 95),
            latency_p99=np.percentile(latencies_arr, 99)
        )

        self.results.append(result)
        return result

    def benchmark_end_to_end_pipeline(self, n_iterations: int = 100_000) -> BenchmarkResult:
        """
        测试端到端交易流水线
        数据摄入 -> 特征计算 -> 订单生成 -> 执行
        """
        print(f"\n🔄 Benchmarking End-to-End Pipeline ({n_iterations:,} iterations)...")

        # 初始化组件
        orderbook = cpp_trading2.FastOrderBook()
        executor = cpp_trading2.OrderExecutor()

        # 初始化订单簿
        for i in range(100):
            orderbook.add_bid(100.0 - i * 0.01, 100.0)
            orderbook.add_ask(100.0 + i * 0.01, 100.0)

        # 简单的 JAX 策略函数
        @jit
        def compute_signal(price_window, volume_window):
            """计算交易信号"""
            ma_fast = jnp.mean(price_window[-5:])
            ma_slow = jnp.mean(price_window)
            vol_ratio = volume_window[-1] / jnp.mean(volume_window)
            signal = (ma_fast - ma_slow) * vol_ratio
            return signal

        # 生成测试数据
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n_iterations) * 0.01)
        volumes = np.random.uniform(100.0, 1000.0, n_iterations)

        price_window = jnp.array(prices[:20])
        volume_window = jnp.array(volumes[:20])

        # 预热
        for i in range(100):
            _ = compute_signal(price_window, volume_window)

        latencies = []
        start = time.perf_counter()

        for i in range(20, n_iterations):
            t0 = time.perf_counter()

            # 1. 数据摄入（存储到内存）
            current_tick = {
                'price': prices[i],
                'volume': volumes[i],
                'timestamp': float(i)
            }

            # 2. 更新窗口
            price_window = jnp.concatenate([price_window[1:], jnp.array([prices[i]])])
            volume_window = jnp.concatenate([volume_window[1:], jnp.array([volumes[i]])])

            # 3. 计算信号
            signal = compute_signal(price_window, volume_window)
            signal_value = float(signal)

            # 4. 生成并执行订单（如果信号足够强）
            if abs(signal_value) > 0.1:
                side = cpp_trading2.OrderSide.BUY if signal_value > 0 else cpp_trading2.OrderSide.SELL

                # 5. 提交订单
                executor.submit_order(side, cpp_trading2.HFTOrderType.MARKET, 10.0, prices[i])

                # 6. 执行待处理订单
                executor.execute_pending_orders(orderbook)

            latencies.append((time.perf_counter() - t0) * 1e6)

        elapsed = time.perf_counter() - start
        throughput = (n_iterations - 20) / elapsed

        latencies_arr = np.array(latencies)
        result = BenchmarkResult(
            name="End-to-End Pipeline",
            data_points=n_iterations - 20,
            elapsed_time=elapsed,
            throughput=throughput,
            latency_mean=latencies_arr.mean(),
            latency_p50=np.percentile(latencies_arr, 50),
            latency_p95=np.percentile(latencies_arr, 95),
            latency_p99=np.percentile(latencies_arr, 99)
        )

        self.results.append(result)
        return result

    def print_summary(self):
        """打印性能测试总结"""
        print("\n" + "="*80)
        print("🎯 THROUGHPUT BENCHMARK SUMMARY")
        print("="*80)

        for result in self.results:
            print(f"\n📌 {result.name}")
            print(f"   Data Points:     {result.data_points:>15,}")
            print(f"   Total Time:      {result.elapsed_time:>15.3f} seconds")
            print(f"   Throughput:      {result.throughput:>15,.0f} ops/sec")
            print(f"   Latency (mean):  {result.latency_mean:>15.2f} μs")
            print(f"   Latency (p50):   {result.latency_p50:>15.2f} μs")
            print(f"   Latency (p95):   {result.latency_p95:>15.2f} μs")
            print(f"   Latency (p99):   {result.latency_p99:>15.2f} μs")

        print("\n" + "="*80)
        print("✅ Benchmark Complete!")
        print("="*80 + "\n")

    def save_results(self, filename: str = None):
        """保存结果到 CSV"""
        import os
        if filename is None:
            # Get script directory and construct path to results
            script_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(os.path.dirname(script_dir), 'results', 'benchmarks')
            filename = os.path.join(results_dir, 'benchmark_results.csv')

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame([
            {
                'Test Name': r.name,
                'Data Points': r.data_points,
                'Elapsed Time (s)': r.elapsed_time,
                'Throughput (ops/s)': r.throughput,
                'Latency Mean (μs)': r.latency_mean,
                'Latency P50 (μs)': r.latency_p50,
                'Latency P95 (μs)': r.latency_p95,
                'Latency P99 (μs)': r.latency_p99,
            }
            for r in self.results
        ])
        df.to_csv(filename, index=False)
        print(f"📊 Results saved to {filename}")


def main():
    """运行所有基准测试"""
    print("🚀 Starting Throughput Benchmarks...")
    print(f"   JAX version: {jax.__version__}")
    print(f"   NumPy version: {np.__version__}")
    print(f"   Available devices: {jax.devices()}")

    benchmark = ThroughputBenchmark()

    # 运行所有测试
    benchmark.benchmark_market_data_ingestion(n_ticks=5_000_000)
    benchmark.benchmark_orderbook_updates(n_updates=1_000_000)
    benchmark.benchmark_order_execution(n_orders=500_000)
    benchmark.benchmark_jax_feature_computation(n_samples=10_000_000)
    benchmark.benchmark_end_to_end_pipeline(n_iterations=100_000)

    # 打印总结
    benchmark.print_summary()

    # 保存结果
    benchmark.save_results()


if __name__ == "__main__":
    main()

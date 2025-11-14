/**
 * CUDA Slippage计算Kernel
 * 
 * 用于批量并行计算slippage，充分利用GPU的并行计算能力
 * 在高频交易场景中，可以同时处理数千个订单的slippage计算
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA Kernel: 批量计算slippage
 * 
 * @param prices 价格数组 [n_orders]
 * @param quantities 数量数组 [n_orders]
 * @param mid_prices 中间价数组 [n_orders]
 * @param sides 买卖方向数组 [n_orders] (1.0=buy, -1.0=sell)
 * @param slippage_costs 输出的slippage成本数组 [n_orders]
 * @param n_orders 订单数量
 * @param base_slippage_bps 基础slippage (basis points)
 * @param volatility 波动率调整因子
 * @param liquidity_factor 流动性因子
 * @param size_threshold 订单规模阈值
 */
__global__ void calculate_slippage_batch_kernel(
    const float* __restrict__ prices,
    const float* __restrict__ quantities,
    const float* __restrict__ mid_prices,
    const float* __restrict__ sides,
    float* __restrict__ slippage_costs,
    const int n_orders,
    const float base_slippage_bps,
    const float volatility,
    const float liquidity_factor,
    const float size_threshold
) {
    // 每个线程处理一个订单
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_orders) return;
    
    // 转换为basis points
    const float base_slip = base_slippage_bps / 10000.0f;
    
    // 预计算调整因子
    const float vol_adjustment = volatility * 0.5f;
    const float liquidity_adjustment = (2.0f - liquidity_factor) * 0.5f;
    
    // Order size adjustment
    const float notional_value = quantities[idx] * prices[idx];
    const float size_adjustment = fminf(notional_value / size_threshold, 2.0f);
    
    // Total slippage factor
    const float total_slippage_factor = base_slip * (1.0f + vol_adjustment + liquidity_adjustment + size_adjustment);
    
    // Calculate slippage
    const float slippage = prices[idx] * total_slippage_factor;
    
    // Apply direction
    const float execution_price = mid_prices[idx] + (sides[idx] * slippage);
    
    // Calculate total slippage cost
    slippage_costs[idx] = fabsf(execution_price - mid_prices[idx]) * quantities[idx];
}

/**
 * 优化的CUDA Kernel: 使用共享内存和warp-level优化
 */
__global__ void calculate_slippage_batch_optimized_kernel(
    const float* __restrict__ prices,
    const float* __restrict__ quantities,
    const float* __restrict__ mid_prices,
    const float* __restrict__ sides,
    float* __restrict__ slippage_costs,
    const int n_orders,
    const float base_slippage_bps,
    const float volatility,
    const float liquidity_factor,
    const float size_threshold
) {
    // 使用共享内存缓存参数（虽然这里参数很少，但展示了优化思路）
    __shared__ float shared_params[4];
    
    if (threadIdx.x == 0) {
        shared_params[0] = base_slippage_bps / 10000.0f;
        shared_params[1] = volatility * 0.5f;
        shared_params[2] = (2.0f - liquidity_factor) * 0.5f;
        shared_params[3] = size_threshold;
    }
    __syncthreads();
    
    const float base_slip = shared_params[0];
    const float vol_adjustment = shared_params[1];
    const float liquidity_adjustment = shared_params[2];
    const float threshold = shared_params[3];
    
    // 每个线程处理一个订单
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_orders) return;
    
    // 使用向量化加载（如果数据对齐）
    const float price = prices[idx];
    const float quantity = quantities[idx];
    const float mid_price = mid_prices[idx];
    const float side = sides[idx];
    
    // Order size adjustment
    const float notional_value = quantity * price;
    const float size_adjustment = fminf(notional_value / threshold, 2.0f);
    
    // Total slippage factor
    const float total_slippage_factor = base_slip * (1.0f + vol_adjustment + liquidity_adjustment + size_adjustment);
    
    // Calculate slippage
    const float slippage = price * total_slippage_factor;
    
    // Apply direction
    const float execution_price = mid_price + (side * slippage);
    
    // Calculate total slippage cost
    slippage_costs[idx] = fabsf(execution_price - mid_price) * quantity;
}

/**
 * Host函数: 调用CUDA kernel计算slippage
 */
extern "C" void calculate_slippage_batch_cuda(
    const float* prices,
    const float* quantities,
    const float* mid_prices,
    const float* sides,
    float* slippage_costs,
    int n_orders,
    float base_slippage_bps,
    float volatility,
    float liquidity_factor,
    float size_threshold,
    bool use_optimized = true
) {
    // 设置线程块大小（每个block 256个线程）
    const int block_size = 256;
    const int num_blocks = (n_orders + block_size - 1) / block_size;
    
    if (use_optimized) {
        calculate_slippage_batch_optimized_kernel<<<num_blocks, block_size>>>(
            prices, quantities, mid_prices, sides, slippage_costs,
            n_orders, base_slippage_bps, volatility, liquidity_factor, size_threshold
        );
    } else {
        calculate_slippage_batch_kernel<<<num_blocks, block_size>>>(
            prices, quantities, mid_prices, sides, slippage_costs,
            n_orders, base_slippage_bps, volatility, liquidity_factor, size_threshold
        );
    }
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 错误处理（在实际应用中应该更完善）
        return;
    }
    
    // 同步等待kernel完成
    cudaDeviceSynchronize();
}








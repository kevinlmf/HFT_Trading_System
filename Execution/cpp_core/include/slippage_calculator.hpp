#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace hft {

/**
 * 优化的Slippage计算器
 * 用于批量计算slippage，比单个订单计算更高效
 */
class SlippageCalculator {
public:
    struct SlippageParams {
        double base_slippage_bps;      // 基础slippage (basis points)
        double volatility;             // 波动率调整因子
        double liquidity_factor;       // 流动性因子
        double size_threshold;         // 订单规模阈值

        SlippageParams(
            double base_slippage = 1.0,
            double vol = 1.0,
            double liquidity = 1.0,
            double threshold = 100000.0
        )
            : base_slippage_bps(base_slippage),
              volatility(vol),
              liquidity_factor(liquidity),
              size_threshold(threshold) {}
    };

    SlippageCalculator(const SlippageParams& params = SlippageParams())
        : params_(params) {}

    /**
     * 批量计算slippage
     * 
     * @param prices 价格数组
     * @param quantities 数量数组
     * @param mid_prices 中间价数组
     * @param sides 买卖方向数组 (1=buy, -1=sell)
     * @param slippage_costs 输出的slippage成本数组
     */
    void calculate_batch(
        const std::vector<double>& prices,
        const std::vector<double>& quantities,
        const std::vector<double>& mid_prices,
        const std::vector<int>& sides,
        std::vector<double>& slippage_costs
    ) {
        size_t n = prices.size();
        slippage_costs.resize(n);
        
        // 转换为basis points
        double base_slip = params_.base_slippage_bps / 10000.0;
        
        // 预计算调整因子
        double vol_adjustment = params_.volatility * 0.5;
        double liquidity_adjustment = (2.0 - params_.liquidity_factor) * 0.5;
        
        // 批量计算
        for (size_t i = 0; i < n; ++i) {
            // Order size adjustment
            double notional_value = quantities[i] * prices[i];
            double size_adjustment = std::min(notional_value / params_.size_threshold, 2.0);
            
            // Total slippage factor
            double total_slippage_factor = base_slip * (1.0 + vol_adjustment + liquidity_adjustment + size_adjustment);
            
            // Calculate slippage
            double slippage = prices[i] * total_slippage_factor;
            
            // Apply direction
            double execution_price = mid_prices[i] + (sides[i] * slippage);
            
            // Calculate total slippage cost
            slippage_costs[i] = std::abs(execution_price - mid_prices[i]) * quantities[i];
        }
    }

    /**
     * 单订单slippage计算（用于兼容性）
     */
    double calculate_single(
        double price,
        double quantity,
        double mid_price,
        int side
    ) {
        double base_slip = params_.base_slippage_bps / 10000.0;
        double vol_adjustment = params_.volatility * 0.5;
        double liquidity_adjustment = (2.0 - params_.liquidity_factor) * 0.5;
        
        double notional_value = quantity * price;
        double size_adjustment = std::min(notional_value / params_.size_threshold, 2.0);
        
        double total_slippage_factor = base_slip * (1.0 + vol_adjustment + liquidity_adjustment + size_adjustment);
        double slippage = price * total_slippage_factor;
        
        double execution_price = mid_price + (side * slippage);
        return std::abs(execution_price - mid_price) * quantity;
    }

    void set_params(const SlippageParams& params) {
        params_ = params;
    }

    const SlippageParams& get_params() const {
        return params_;
    }

private:
    SlippageParams params_;
};

}  // namespace hft



#pragma once
#include "fast_orderbook.hpp"
#include <vector>
#include <chrono>
#include <cmath>

namespace hft {

enum class OrderSide : int8_t {
    BUY = 1,
    SELL = -1
};

enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1
};

struct Order {
    uint64_t order_id;
    OrderSide side;
    OrderType type;
    double quantity;
    double price;  // For limit orders
    uint64_t timestamp_ns;

    Order() : order_id(0), side(OrderSide::BUY), type(OrderType::MARKET),
              quantity(0), price(0), timestamp_ns(0) {}
};

struct Fill {
    uint64_t order_id;
    double executed_qty;
    double avg_price;
    double commission;
    double slippage;
    uint64_t timestamp_ns;

    Fill() : order_id(0), executed_qty(0), avg_price(0),
             commission(0), slippage(0), timestamp_ns(0) {}
};

// Ultra-fast order executor
class OrderExecutor {
public:
    OrderExecutor(double commission_rate = 0.0001,  // 0.01%
                 double min_commission = 0.0)
        : commission_rate_(commission_rate),
          min_commission_(min_commission),
          next_order_id_(1) {}

    // ========== Submit Order ==========
    uint64_t submit_order(OrderSide side, OrderType type, double quantity, double price = 0.0) {
        Order order;
        order.order_id = next_order_id_++;
        order.side = side;
        order.type = type;
        order.quantity = quantity;
        order.price = price;
        order.timestamp_ns = get_timestamp_ns();

        pending_orders_.push_back(order);
        return order.order_id;
    }

    // ========== Execute Against OrderBook ==========
    std::vector<Fill> execute_pending_orders(FastOrderBook& book) {
        std::vector<Fill> fills;
        fills.reserve(pending_orders_.size());

        for (const auto& order : pending_orders_) {
            Fill fill = execute_single_order(order, book);
            if (fill.executed_qty > 0) {
                fills.push_back(fill);
                fill_history_.push_back(fill);
            }
        }

        pending_orders_.clear();
        return fills;
    }

    // ========== Execute Single Order ==========
    Fill execute_single_order(const Order& order, FastOrderBook& book) {
        Fill fill;
        fill.order_id = order.order_id;
        fill.timestamp_ns = get_timestamp_ns();

        double executed_qty = 0.0;
        double avg_price = 0.0;

        if (order.type == OrderType::MARKET) {
            // Market order execution
            if (order.side == OrderSide::BUY) {
                auto [qty, price] = book.execute_market_buy(order.quantity);
                executed_qty = qty;
                avg_price = price;
            } else {
                auto [qty, price] = book.execute_market_sell(order.quantity);
                executed_qty = qty;
                avg_price = price;
            }
        } else {
            // Limit order (simplified - immediate or nothing)
            if (order.side == OrderSide::BUY && order.price >= book.best_ask()) {
                auto [qty, price] = book.execute_market_buy(order.quantity);
                executed_qty = qty;
                avg_price = price;
            } else if (order.side == OrderSide::SELL && order.price <= book.best_bid()) {
                auto [qty, price] = book.execute_market_sell(order.quantity);
                executed_qty = qty;
                avg_price = price;
            }
        }

        fill.executed_qty = executed_qty;
        fill.avg_price = avg_price;

        if (executed_qty > 0) {
            // Calculate slippage
            double mid = book.mid_price();
            fill.slippage = std::abs(avg_price - mid);

            // Calculate commission
            double notional = executed_qty * avg_price;
            fill.commission = std::max(notional * commission_rate_, min_commission_);
        }

        return fill;
    }

    // ========== Statistics ==========
    size_t get_fill_count() const { return fill_history_.size(); }

    double get_total_commission() const {
        double total = 0.0;
        for (const auto& fill : fill_history_) {
            total += fill.commission;
        }
        return total;
    }

    double get_total_slippage() const {
        double total = 0.0;
        for (const auto& fill : fill_history_) {
            total += fill.slippage * fill.executed_qty;
        }
        return total;
    }

    void reset() {
        pending_orders_.clear();
        fill_history_.clear();
        next_order_id_ = 1;
    }

    // ========== Access Fills (for JAX) ==========
    const std::vector<Fill>& get_fill_history() const { return fill_history_; }

private:
    uint64_t get_timestamp_ns() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }

    double commission_rate_;
    double min_commission_;
    uint64_t next_order_id_;

    std::vector<Order> pending_orders_;
    std::vector<Fill> fill_history_;
};

}  // namespace hft

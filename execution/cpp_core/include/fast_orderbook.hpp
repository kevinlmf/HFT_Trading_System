#pragma once
#include <array>
#include <cstdint>
#include <algorithm>
#include <cstring>

// Ultra-fast OrderBook for HFT
// Design goals: < 5µs latency, zero-copy for jaxbind

namespace hft {

constexpr size_t MAX_PRICE_LEVELS = 1000;

struct PriceLevel {
    double price;
    double quantity;

    PriceLevel() : price(0.0), quantity(0.0) {}
    PriceLevel(double p, double q) : price(p), quantity(q) {}
};

// Lock-free array-based order book
class FastOrderBook {
public:
    FastOrderBook() {
        reset();
    }

    void reset() {
        bid_count_ = 0;
        ask_count_ = 0;
        memset(bids_.data(), 0, sizeof(PriceLevel) * MAX_PRICE_LEVELS);
        memset(asks_.data(), 0, sizeof(PriceLevel) * MAX_PRICE_LEVELS);
    }

    // ========== Add Liquidity ==========
    void add_bid(double price, double qty) {
        if (bid_count_ < MAX_PRICE_LEVELS && qty > 0) {
            bids_[bid_count_++] = PriceLevel{price, qty};
        }
    }

    void add_ask(double price, double qty) {
        if (ask_count_ < MAX_PRICE_LEVELS && qty > 0) {
            asks_[ask_count_++] = PriceLevel{price, qty};
        }
    }

    // ========== Market Data ==========
    double best_bid() const {
        if (bid_count_ == 0) return 0.0;
        double max_price = bids_[0].price;
        for (size_t i = 1; i < bid_count_; ++i) {
            if (bids_[i].quantity > 0) {
                max_price = std::max(max_price, bids_[i].price);
            }
        }
        return max_price;
    }

    double best_ask() const {
        if (ask_count_ == 0) return 0.0;
        double min_price = asks_[0].price;
        for (size_t i = 1; i < ask_count_; ++i) {
            if (asks_[i].quantity > 0) {
                min_price = std::min(min_price, asks_[i].price);
            }
        }
        return min_price;
    }

    double mid_price() const {
        double bid = best_bid();
        double ask = best_ask();
        if (bid == 0.0 && ask == 0.0) return 0.0;
        if (bid == 0.0) return ask;
        if (ask == 0.0) return bid;
        return (bid + ask) / 2.0;
    }

    double spread() const {
        double bid = best_bid();
        double ask = best_ask();
        return (ask > bid) ? (ask - bid) : 0.0;
    }

    // ========== Order Execution ==========
    // Returns: [executed_qty, avg_price]
    std::pair<double, double> execute_market_buy(double quantity) {
        double remaining = quantity;
        double total_cost = 0.0;
        double executed = 0.0;

        for (size_t i = 0; i < ask_count_ && remaining > 0; ++i) {
            if (asks_[i].quantity > 0) {
                double exec_qty = std::min(remaining, asks_[i].quantity);
                total_cost += exec_qty * asks_[i].price;
                asks_[i].quantity -= exec_qty;
                remaining -= exec_qty;
                executed += exec_qty;
            }
        }

        double avg_price = (executed > 0) ? (total_cost / executed) : 0.0;
        return {executed, avg_price};
    }

    std::pair<double, double> execute_market_sell(double quantity) {
        double remaining = quantity;
        double total_value = 0.0;
        double executed = 0.0;

        for (size_t i = 0; i < bid_count_ && remaining > 0; ++i) {
            if (bids_[i].quantity > 0) {
                double exec_qty = std::min(remaining, bids_[i].quantity);
                total_value += exec_qty * bids_[i].price;
                bids_[i].quantity -= exec_qty;
                remaining -= exec_qty;
                executed += exec_qty;
            }
        }

        double avg_price = (executed > 0) ? (total_value / executed) : 0.0;
        return {executed, avg_price};
    }

    // ========== Zero-Copy Export for JAX ==========
    const double* get_bid_prices_ptr() const {
        return reinterpret_cast<const double*>(bids_.data());
    }

    const double* get_bid_quantities_ptr() const {
        return reinterpret_cast<const double*>(bids_.data()) + 1;
    }

    const double* get_ask_prices_ptr() const {
        return reinterpret_cast<const double*>(asks_.data());
    }

    const double* get_ask_quantities_ptr() const {
        return reinterpret_cast<const double*>(asks_.data()) + 1;
    }

    size_t get_bid_count() const { return bid_count_; }
    size_t get_ask_count() const { return ask_count_; }

    // ========== Batch Update from JAX ==========
    void update_from_arrays(const double* bid_prices, const double* bid_qtys, size_t n_bids,
                           const double* ask_prices, const double* ask_qtys, size_t n_asks) {
        bid_count_ = std::min(n_bids, MAX_PRICE_LEVELS);
        ask_count_ = std::min(n_asks, MAX_PRICE_LEVELS);

        for (size_t i = 0; i < bid_count_; ++i) {
            bids_[i] = PriceLevel{bid_prices[i], bid_qtys[i]};
        }

        for (size_t i = 0; i < ask_count_; ++i) {
            asks_[i] = PriceLevel{ask_prices[i], ask_qtys[i]};
        }
    }

    // ========== Book Depth Analysis ==========
    double total_bid_liquidity(size_t depth = 10) const {
        double total = 0.0;
        size_t limit = std::min(depth, bid_count_);
        for (size_t i = 0; i < limit; ++i) {
            total += bids_[i].quantity;
        }
        return total;
    }

    double total_ask_liquidity(size_t depth = 10) const {
        double total = 0.0;
        size_t limit = std::min(depth, ask_count_);
        for (size_t i = 0; i < limit; ++i) {
            total += asks_[i].quantity;
        }
        return total;
    }

    // Order Book Imbalance (OBI)
    double order_book_imbalance() const {
        double bid_liq = total_bid_liquidity();
        double ask_liq = total_ask_liquidity();
        double total = bid_liq + ask_liq;
        return (total > 0) ? ((bid_liq - ask_liq) / total) : 0.0;
    }

private:
    std::array<PriceLevel, MAX_PRICE_LEVELS> bids_;
    std::array<PriceLevel, MAX_PRICE_LEVELS> asks_;
    size_t bid_count_;
    size_t ask_count_;
};

}  // namespace hft

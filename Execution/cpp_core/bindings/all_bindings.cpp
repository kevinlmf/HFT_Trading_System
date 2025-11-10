#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "data_feed.h"
#include "order.hpp"
#include "fast_orderbook.hpp"
#include "order_executor.hpp"
#include "slippage_calculator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_trading2, m) {
    // --- Bind DataFeed ---
    py::class_<Row>(m, "Row")
        .def_readwrite("date", &Row::date)
        .def_readwrite("open", &Row::open)
        .def_readwrite("high", &Row::high)
        .def_readwrite("low", &Row::low)
        .def_readwrite("close", &Row::close)
        .def_readwrite("volume", &Row::volume);

    py::class_<DataFeed>(m, "DataFeed")
        .def(py::init<>())
        .def("load", &DataFeed::load)
        .def("next", &DataFeed::next)
        .def("current", &DataFeed::current)
        .def("moving_average", &DataFeed::moving_average);

    // --- Bind OrderType enum ---
    py::enum_<OrderType>(m, "OrderType")
        .value("BUY", OrderType::BUY)
        .value("SELL", OrderType::SELL);

    // --- Bind Order struct ---
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("type", &Order::type)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("timestamp", &Order::timestamp);

    // --- Bind HFT OrderSide enum ---
    py::enum_<hft::OrderSide>(m, "OrderSide")
        .value("BUY", hft::OrderSide::BUY)
        .value("SELL", hft::OrderSide::SELL);

    // --- Bind HFT OrderType enum ---
    py::enum_<hft::OrderType>(m, "HFTOrderType")
        .value("MARKET", hft::OrderType::MARKET)
        .value("LIMIT", hft::OrderType::LIMIT);

    // --- Bind HFT Order struct ---
    py::class_<hft::Order>(m, "HFTOrder")
        .def(py::init<>())
        .def_readwrite("order_id", &hft::Order::order_id)
        .def_readwrite("side", &hft::Order::side)
        .def_readwrite("type", &hft::Order::type)
        .def_readwrite("quantity", &hft::Order::quantity)
        .def_readwrite("price", &hft::Order::price)
        .def_readwrite("timestamp_ns", &hft::Order::timestamp_ns);

    // --- Bind HFT Fill struct ---
    py::class_<hft::Fill>(m, "Fill")
        .def(py::init<>())
        .def_readwrite("order_id", &hft::Fill::order_id)
        .def_readwrite("executed_qty", &hft::Fill::executed_qty)
        .def_readwrite("avg_price", &hft::Fill::avg_price)
        .def_readwrite("commission", &hft::Fill::commission)
        .def_readwrite("slippage", &hft::Fill::slippage)
        .def_readwrite("timestamp_ns", &hft::Fill::timestamp_ns);

    // --- Bind FastOrderBook class ---
    py::class_<hft::FastOrderBook>(m, "FastOrderBook")
        .def(py::init<>())
        .def("reset", &hft::FastOrderBook::reset)
        .def("add_bid", &hft::FastOrderBook::add_bid)
        .def("add_ask", &hft::FastOrderBook::add_ask)
        .def("best_bid", &hft::FastOrderBook::best_bid)
        .def("best_ask", &hft::FastOrderBook::best_ask)
        .def("mid_price", &hft::FastOrderBook::mid_price)
        .def("spread", &hft::FastOrderBook::spread)
        .def("execute_market_buy", &hft::FastOrderBook::execute_market_buy)
        .def("execute_market_sell", &hft::FastOrderBook::execute_market_sell)
        .def("get_bid_count", &hft::FastOrderBook::get_bid_count)
        .def("get_ask_count", &hft::FastOrderBook::get_ask_count)
        .def("total_bid_liquidity", &hft::FastOrderBook::total_bid_liquidity, py::arg("depth") = 10)
        .def("total_ask_liquidity", &hft::FastOrderBook::total_ask_liquidity, py::arg("depth") = 10)
        .def("order_book_imbalance", &hft::FastOrderBook::order_book_imbalance)
        .def("update_from_arrays", &hft::FastOrderBook::update_from_arrays);

    // --- Bind OrderExecutor class ---
    py::class_<hft::OrderExecutor>(m, "OrderExecutor")
        .def(py::init<>())
        .def("submit_order", &hft::OrderExecutor::submit_order)
        .def("execute_pending_orders", &hft::OrderExecutor::execute_pending_orders)
        .def("get_fill_history", &hft::OrderExecutor::get_fill_history)
        .def("get_fill_count", &hft::OrderExecutor::get_fill_count)
        .def("get_total_commission", &hft::OrderExecutor::get_total_commission)
        .def("get_total_slippage", &hft::OrderExecutor::get_total_slippage)
        .def("reset", &hft::OrderExecutor::reset);

    // --- Bind SlippageCalculator ---
    py::class_<hft::SlippageCalculator::SlippageParams>(m, "SlippageParams")
        .def(py::init<>())
        .def_readwrite("base_slippage_bps", &hft::SlippageCalculator::SlippageParams::base_slippage_bps)
        .def_readwrite("volatility", &hft::SlippageCalculator::SlippageParams::volatility)
        .def_readwrite("liquidity_factor", &hft::SlippageCalculator::SlippageParams::liquidity_factor)
        .def_readwrite("size_threshold", &hft::SlippageCalculator::SlippageParams::size_threshold);

    py::class_<hft::SlippageCalculator>(m, "SlippageCalculator")
        .def(py::init<>())
        .def(py::init<const hft::SlippageCalculator::SlippageParams&>())
        .def("calculate_batch", [](hft::SlippageCalculator& calc,
                                   py::array_t<double> prices,
                                   py::array_t<double> quantities,
                                   py::array_t<double> mid_prices,
                                   py::array_t<int> sides) {
            auto prices_buf = prices.request();
            auto quantities_buf = quantities.request();
            auto mid_prices_buf = mid_prices.request();
            auto sides_buf = sides.request();
            
            size_t n = prices_buf.size;
            if (quantities_buf.size != n || mid_prices_buf.size != n || sides_buf.size != n) {
                throw std::runtime_error("All arrays must have the same size");
            }
            
            std::vector<double> prices_vec(static_cast<double*>(prices_buf.ptr),
                                          static_cast<double*>(prices_buf.ptr) + n);
            std::vector<double> quantities_vec(static_cast<double*>(quantities_buf.ptr),
                                             static_cast<double*>(quantities_buf.ptr) + n);
            std::vector<double> mid_prices_vec(static_cast<double*>(mid_prices_buf.ptr),
                                              static_cast<double*>(mid_prices_buf.ptr) + n);
            std::vector<int> sides_vec(static_cast<int*>(sides_buf.ptr),
                                      static_cast<int*>(sides_buf.ptr) + n);
            
            std::vector<double> slippage_costs;
            calc.calculate_batch(prices_vec, quantities_vec, mid_prices_vec, sides_vec, slippage_costs);
            
            return py::array_t<double>(slippage_costs.size(), slippage_costs.data());
        })
        .def("calculate_single", &hft::SlippageCalculator::calculate_single)
        .def("set_params", &hft::SlippageCalculator::set_params)
        .def("get_params", &hft::SlippageCalculator::get_params);
}

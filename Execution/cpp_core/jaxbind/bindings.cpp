#include <jaxbind/jaxbind.h>
#include "../include/fast_orderbook.hpp"
#include "../include/order_executor.hpp"

namespace jb = jaxbind;

// ========== FastOrderBook Bindings ==========
JAXBIND_MODULE(hft_cpp) {
    using namespace hft;

    // OrderBook class
    jb::Class<FastOrderBook>(m, "FastOrderBook")
        .def(jb::init<>())
        .def("reset", &FastOrderBook::reset)
        .def("add_bid", &FastOrderBook::add_bid)
        .def("add_ask", &FastOrderBook::add_ask)
        .def("best_bid", &FastOrderBook::best_bid)
        .def("best_ask", &FastOrderBook::best_ask)
        .def("mid_price", &FastOrderBook::mid_price)
        .def("spread", &FastOrderBook::spread)
        .def("execute_market_buy", &FastOrderBook::execute_market_buy)
        .def("execute_market_sell", &FastOrderBook::execute_market_sell)
        .def("get_bid_count", &FastOrderBook::get_bid_count)
        .def("get_ask_count", &FastOrderBook::get_ask_count)
        .def("total_bid_liquidity", &FastOrderBook::total_bid_liquidity, jb::arg("depth") = 10)
        .def("total_ask_liquidity", &FastOrderBook::total_ask_liquidity, jb::arg("depth") = 10)
        .def("order_book_imbalance", &FastOrderBook::order_book_imbalance)
        .def("update_from_arrays",
             [](FastOrderBook& self,
                jb::ArrayRef<const double> bid_prices,
                jb::ArrayRef<const double> bid_qtys,
                jb::ArrayRef<const double> ask_prices,
                jb::ArrayRef<const double> ask_qtys) {
                 self.update_from_arrays(
                     bid_prices.data(), bid_qtys.data(), bid_prices.size(),
                     ask_prices.data(), ask_qtys.data(), ask_prices.size()
                 );
             })
        // Zero-copy access to internal arrays
        .def("get_bid_data",
             [](const FastOrderBook& self) {
                 size_t n = self.get_bid_count();
                 return jb::Array<double>({n, 2}, self.get_bid_prices_ptr());
             })
        .def("get_ask_data",
             [](const FastOrderBook& self) {
                 size_t n = self.get_ask_count();
                 return jb::Array<double>({n, 2}, self.get_ask_prices_ptr());
             });

    // OrderSide enum
    jb::Enum<OrderSide>(m, "OrderSide")
        .value("BUY", OrderSide::BUY)
        .value("SELL", OrderSide::SELL);

    // OrderType enum
    jb::Enum<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::MARKET)
        .value("LIMIT", OrderType::LIMIT);

    // Fill struct (read-only from Python)
    jb::Class<Fill>(m, "Fill")
        .def_readonly("order_id", &Fill::order_id)
        .def_readonly("executed_qty", &Fill::executed_qty)
        .def_readonly("avg_price", &Fill::avg_price)
        .def_readonly("commission", &Fill::commission)
        .def_readonly("slippage", &Fill::slippage)
        .def_readonly("timestamp_ns", &Fill::timestamp_ns);

    // OrderExecutor class
    jb::Class<OrderExecutor>(m, "OrderExecutor")
        .def(jb::init<double, double, double>(),
             jb::arg("commission_rate") = 0.0001,
             jb::arg("min_commission") = 0.0,
             jb::arg("slippage_bps") = 1.0)
        .def("submit_order", &OrderExecutor::submit_order,
             jb::arg("side"),
             jb::arg("type"),
             jb::arg("quantity"),
             jb::arg("price") = 0.0)
        .def("execute_pending_orders", &OrderExecutor::execute_pending_orders)
        .def("get_fill_count", &OrderExecutor::get_fill_count)
        .def("get_total_commission", &OrderExecutor::get_total_commission)
        .def("get_total_slippage", &OrderExecutor::get_total_slippage)
        .def("reset", &OrderExecutor::reset)
        .def("get_fill_history",
             [](const OrderExecutor& self) {
                 const auto& fills = self.get_fill_history();
                 return fills;  // jaxbind will handle vector conversion
             });

    // Utility functions
    m.def("create_synthetic_book",
          [](double mid_price, size_t levels) -> FastOrderBook {
              FastOrderBook book;
              double tick = 0.01;
              for (size_t i = 0; i < levels; ++i) {
                  book.add_bid(mid_price - (i + 1) * tick, 100.0 - i * 5);
                  book.add_ask(mid_price + (i + 1) * tick, 100.0 - i * 5);
              }
              return book;
          },
          jb::arg("mid_price") = 100.0,
          jb::arg("levels") = 10);
}

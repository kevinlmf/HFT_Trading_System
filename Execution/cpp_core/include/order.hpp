#pragma once
#include <string>

enum class OrderType { BUY, SELL };

struct Order {
    std::string symbol;
    OrderType type;
    double price;
    int quantity;
    long timestamp;
};

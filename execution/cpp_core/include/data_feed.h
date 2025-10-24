// cpp_core/include/data_feed.h
#pragma once
#include <string>
#include <vector>

struct Row {
    std::string date;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
};

class DataFeed {
public:
    bool load(const std::string& path);
    bool next();
    Row current() const;
    std::vector<double> moving_average(int window) const;

private:
    std::vector<Row> data_;
    int current_index_ = -1;
};

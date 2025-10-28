#include "data_feed.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>

bool DataFeed::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    std::string line;

    // ✅ 跳过前两行（Header + Ticker row）
    std::getline(file, line);
    std::getline(file, line);

    data_.clear();
    current_index_ = -1;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        Row row;

        try {
            std::getline(ss, row.date, ',');

            std::getline(ss, token, ',');
            row.open = std::stod(token);

            std::getline(ss, token, ',');
            row.high = std::stod(token);

            std::getline(ss, token, ',');
            row.low = std::stod(token);

            std::getline(ss, token, ',');
            row.close = std::stod(token);

            std::getline(ss, token, ',');
            row.volume = std::stod(token);

            data_.push_back(row);
        } catch (const std::exception& e) {
            std::cerr << "⚠️  Skipping invalid row: " << line << "\nReason: " << e.what() << std::endl;
            continue;
        }
    }

    return !data_.empty();
}

bool DataFeed::next() {
    if (current_index_ + 1 < static_cast<int>(data_.size())) {
        ++current_index_;
        return true;
    }
    return false;
}

Row DataFeed::current() const {
    if (current_index_ >= 0 && current_index_ < static_cast<int>(data_.size())) {
        return data_[current_index_];
    }
    return {};
}

std::vector<double> DataFeed::moving_average(int window) const {
    std::vector<double> result;
    int n = data_.size();
    if (window <= 0 || n < window) return result;

    result.reserve(n - window + 1);
    for (int i = 0; i <= n - window; ++i) {
        double sum = 0.0;
        for (int j = 0; j < window; ++j) {
            sum += data_[i + j].close;
        }
        result.push_back(sum / window);
    }
    return result;
}


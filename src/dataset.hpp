#pragma once

#include <vector>

struct Dataset
{
    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    Dataset(std::vector<std::vector<int>> row_data, std::vector<int> target) : row_data(std::move(row_data)), target_data(std::move(target)) {}

    Dataset() = default;

    size_t num_rows() const { return target_data.size(); }

    size_t num_attributes() const { return row_data[0].size(); }
};

#include "csv.h"
#include "dataset.hpp"
#include "tree.hpp"
#include <gtest/gtest.h>

TEST(TennisTest, AllInSample)
{
    constexpr int cols = 5;

    io::CSVReader<cols> in("../datasets/tennis.csv");
    int outlook, temp, humidity, wind, play;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(outlook, temp, humidity, wind, play))
    {
        row_data.push_back({outlook, temp, humidity, wind});
        target_data.push_back(play);
    }

    // Dataset dataset(std::make_shared<InnerDataset>(row_data, target_data));

    const auto tree = build_tree(row_data, target_data);

    for (int i = 0; i < row_data.size(); ++i)
    {
        EXPECT_EQ(tree_predict(row_data[i], tree), target_data[i]);
    }
}

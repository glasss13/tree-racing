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

    Dataset dataset(std::make_shared<InnerDataset>(row_data, target_data));

    const auto tree = build_tree(dataset);

    for (int i = 0; i < dataset.num_rows(); ++i)
    {
        EXPECT_EQ(tree_predict(dataset.get_row(i), tree), dataset.get_target(i));
    }
}

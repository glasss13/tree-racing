#include "csv.h"
#include "tree.hpp"
#include <gtest/gtest.h>

TEST(TennisTest, AllInSample)
{
    constexpr int cols = 5;

    io::CSVReader<cols> in("../datasets/tennis.csv");
    int outlook, temp, humidity, wind, play;

    Dataset dataset(cols);
    int rows = 0;

    while (in.read_row(outlook, temp, humidity, wind, play))
    {
        dataset[0].push_back(outlook);
        dataset[1].push_back(temp);
        dataset[2].push_back(humidity);
        dataset[3].push_back(wind);
        dataset[4].push_back(play);

        ++rows;
    }

    const auto tree = build_tree(dataset);

    for (int i = 0; i < rows; ++i)
    {
        std::vector<int> obs;
        for (int attribute = 0; attribute < cols - 1; ++attribute)
        {
            obs.push_back(dataset[attribute][i]);
        }
        EXPECT_EQ(tree_predict(obs, tree), dataset.back()[i]);
    }
}

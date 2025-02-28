#include "csv.h"
#include "tree.hpp"
#include <gtest/gtest.h>

TEST(CarTest, AllInSample)
{
    constexpr int cols = 7;

    io::CSVReader<cols> in("../datasets/car_eval.csv");
    int buying, maint, doors, person, lug_boot, safety, class_;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(buying, maint, doors, person, lug_boot, safety, class_))
    {
        row_data.push_back({buying, maint, doors, person, lug_boot, safety});
        target_data.push_back(class_);
    }

    Dataset dataset(std::make_shared<InnerDataset>(row_data, target_data));

    const auto tree = build_tree(dataset);

    for (int i = 0; i < dataset.num_rows(); ++i)
    {
        EXPECT_EQ(tree_predict(dataset.get_row(i), tree), dataset.get_target(i));
    }
}

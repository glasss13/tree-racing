#include "csv.h"
#include "tree.hpp"
#include <gtest/gtest.h>

TEST(CarTest, AllInSample)
{
    // car_columns = ["Buying", "Maint", "Doors",
    //                "Persons", "Lug_boot", "Safety", "Class"]
    constexpr int cols = 7;

    io::CSVReader<cols> in("../datasets/car_eval.csv");
    int buying, maint, doors, person, lug_boot, safety, class_;

    Dataset dataset(cols);
    int rows = 0;

    while (in.read_row(buying, maint, doors, person, lug_boot, safety, class_))
    {
        dataset[0].push_back(buying);
        dataset[1].push_back(maint);
        dataset[2].push_back(doors);
        dataset[3].push_back(person);
        dataset[4].push_back(lug_boot);
        dataset[5].push_back(safety);
        dataset[6].push_back(class_);

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

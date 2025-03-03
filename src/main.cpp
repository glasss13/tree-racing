#include "csv.h"
#include "dataset.hpp"
#include "tree.hpp"

#include <benchmark/benchmark.h>

int main()
{
    constexpr int cols = 7;

    io::CSVReader<cols> in("/Users/glass/source/tree-racing/datasets/car_eval.csv");
    int f1, f2, f3, f4, f5, f6, target;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(f1, f2, f3, f4, f5, f6, target))
    {
        row_data.push_back({f1, f2, f3, f4, f5, f6});
        target_data.push_back(target);
    }

    auto [train_set, test_set] = train_test_split(row_data, target_data, 0.8);
    auto &[train_data, train_target] = train_set;
    auto &[test_data, test_target] = test_set;

    // for (int i = 0; i < 100; ++i)
    // {
    const auto tree = build_tree(train_data, train_target);
    //     benchmark::DoNotOptimize(tree);
    // }

    int correct = 0;
    for (size_t i = 0; i < test_data.size(); ++i)
    {
        if (tree_predict(test_data[i], tree) == test_target[i])
        {
            ++correct;
        }
    }

    std::cout << "accuracy: " << static_cast<float>(correct) / test_data.size() << '\n';
    return 0;
}

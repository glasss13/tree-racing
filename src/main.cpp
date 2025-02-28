#include "csv.h"
#include "dataset.hpp"
#include "tree.hpp"

#include <benchmark/benchmark.h>

// int Dataset::cnt = 0;

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

    for (int i = 0; i < 100; ++i)
    {
        auto tree = build_tree(row_data, target_data);
        benchmark::DoNotOptimize(tree);
    }

    // std::cout << "cnt: " << Dataset::get_cnt() << '\n';
    //
    // print_tree(build_tree(row_data, target_data));
    // int correct = 0;
    // for (int i = 0; i < row_data.size(); ++i)
    // {
    //     if (tree_predict(row_data[i], build_tree(row_data, target_data)) == target_data[i])
    //     {
    //         ++correct;
    //     }
    // }
    //
    // std::cout << "accuracy: " << static_cast<float>(correct) / row_data.size() << '\n';
    //
    return 0;
}

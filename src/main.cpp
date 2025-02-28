#include "csv.h"
#include "dataset.hpp"
#include "tree.hpp"

#include <benchmark/benchmark.h>

int main()
{
    constexpr int cols = 5;

    io::CSVReader<cols> in("/Users/glass/source/tree-racing/datasets/tennis.csv");
    int f1, f2, f3, f4, f5, f6, target;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(f1, f2, f3, f4, target))
    {
        row_data.push_back({f1, f2, f3, f4});
        target_data.push_back(target);
    }

    Dataset dataset(std::make_shared<InnerDataset>(row_data, target_data));

    for (int i = 0; i < 1000; ++i)
    {
        auto tree = build_tree(dataset);
        benchmark::DoNotOptimize(tree);
    }

    // print_tree(tree);
    // int correct = 0;
    // for (int i = 0; i < dataset.num_rows(); ++i)
    // {
    //     if (tree_predict(dataset.get_row(i), tree) == dataset.get_target(i))
    //     {
    //         ++correct;
    //     }
    // }
    //
    // std::cout << "accuracy: " << static_cast<float>(correct) / row_data.size() << '\n';

    return 0;
}

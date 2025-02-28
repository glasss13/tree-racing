#include "csv.h"
#include "tree.hpp"
#include <benchmark/benchmark.h>

constexpr const char *file = "/Users/glass/source/tree-racing/datasets/car_eval.csv";

static void BM_BuildTree(benchmark::State &state)
{
    constexpr int cols = 7;
    io::CSVReader<cols> in(file);
    int buying, maint, doors, person, lug_boot, safety, class_;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(buying, maint, doors, person, lug_boot, safety, class_))
    {
        row_data.push_back({buying, maint, doors, person, lug_boot, safety});
        target_data.push_back(class_);
    }

    Dataset dataset(row_data, target_data);

    for (auto _ : state)
    {
        auto tree = build_tree(dataset);
        benchmark::DoNotOptimize(tree);
    }
}

BENCHMARK(BM_BuildTree);

static void BM_TreePredict(benchmark::State &state)
{
    constexpr int cols = 7;
    io::CSVReader<cols> in(file);
    int buying, maint, doors, person, lug_boot, safety, class_;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(buying, maint, doors, person, lug_boot, safety, class_))
    {
        row_data.push_back({buying, maint, doors, person, lug_boot, safety});
        target_data.push_back(class_);
    }

    Dataset dataset(row_data, target_data);

    auto tree = build_tree(dataset);

    for (auto _ : state)
    {
        for (int i = 0; i < dataset.num_rows(); ++i)
        {
            auto pred = tree_predict(dataset.row_data[i], tree);
            benchmark::DoNotOptimize(pred);
        }
    }
}

BENCHMARK(BM_TreePredict);

BENCHMARK_MAIN();

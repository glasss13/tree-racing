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

    auto [train_set, _] = train_test_split(row_data, target_data, 0.8);
    auto &[train_data, train_target] = train_set;

    for (auto _ : state)
    {
        auto tree = build_tree(train_data, train_target);
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

    auto [train_set, test_set] = train_test_split(row_data, target_data, 0.8);
    auto &[train_data, train_target] = train_set;
    auto &[test_data, test_target] = test_set;

    auto tree = build_tree(train_data, train_target);

    for (auto _ : state)
    {
        for (const auto &sample : test_data)
        {
            auto pred = tree_predict(sample, tree);
            benchmark::DoNotOptimize(pred);
        }
    }
}

BENCHMARK(BM_TreePredict);

BENCHMARK_MAIN();

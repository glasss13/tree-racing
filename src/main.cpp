#include "csv.h"
#include "dataset.hpp"
#include "tree.hpp"

#include <benchmark/benchmark.h>
#include <chrono>

std::pair<std::vector<std::vector<int>>, std::vector<int>> loadTennisData()
{
    constexpr int cols = 5;

    io::CSVReader<cols> in("datasets/tennis.csv");
    int outlook, temp, humidity, wind, play;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(outlook, temp, humidity, wind, play))
    {
        row_data.push_back({outlook, temp, humidity, wind});
        target_data.push_back(play);
    }
    return {row_data, target_data};
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> loadMushroomData()
{
    io::CSVReader<23> csvReader("datasets/mushroom.csv");

    std::vector<std::vector<int>> features;
    std::vector<int> labels;
    int label;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22;
    while (csvReader.read_row(label, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22))
    {
        features.push_back({f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22});
        labels.push_back(label);
    }
    return {features, labels};
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> loadZooData()
{
    io::CSVReader<17> csvReader("datasets/zoo.csv");

    std::vector<std::vector<int>> features;
    std::vector<int> labels;
    int label;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16;
    while (csvReader.read_row(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, label))
    {
        features.push_back({f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16});
        labels.push_back(label);
    }
    return {features, labels};
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> loadCarData()
{
    constexpr int cols = 7;

    io::CSVReader<cols> in("datasets/car_eval.csv");
    int f1, f2, f3, f4, f5, f6, target;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(f1, f2, f3, f4, f5, f6, target))
    {
        row_data.push_back({f1, f2, f3, f4, f5, f6});
        target_data.push_back(target);
    }

    return {row_data, target_data};
}

int main()
{

    auto [row_data, target_data] = loadCarData();

    float train_time = 0;
    float pred_time = 0;
    int correct = 0;
    int total = 0;
    for (int i = 0; i < 10'000; ++i)
    {
        auto [train_set, test_set] = train_test_split(row_data, target_data, 0.85);
        auto &[train_data, train_target] = train_set;
        auto &[test_data, test_target] = test_set;

        const auto st = std::chrono::high_resolution_clock::now();

        auto tree = build_tree(train_data, train_target);

        const auto end = std::chrono::high_resolution_clock::now();

        train_time += std::chrono::duration<float, std::micro>(end - st).count();

        const auto st2 = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_data.size(); ++i)
        {
            float pred = tree_predict(test_data[i], tree);
            correct += pred == test_target[i];
            ++total;
        }

        const auto end2 = std::chrono::high_resolution_clock::now();

        pred_time += std::chrono::duration<float, std::micro>(end2 - st2).count();
    }

    std::cout << "Train time: " << train_time / 1'000 << " us\n";
    std::cout << "Pred time: " << pred_time / total << " us\n";
    std::cout << "Accuracy: " << static_cast<float>(correct) / total << '\n';
}

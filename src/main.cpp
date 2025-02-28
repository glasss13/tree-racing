#include "csv.h"
#include "tree.hpp"

int main()
{
    constexpr int cols = 5;

    io::CSVReader<cols> in("datasets/tennis.csv");
    int outlook, temp, humidity, wind, play;

    std::vector<std::vector<int>> row_data;
    std::vector<int> target_data;

    while (in.read_row(outlook, temp, humidity, wind, play))
    {
        row_data.push_back({outlook, temp, humidity, wind});
        row_data.push_back(target_data);
        // dataset[0].push_back(outlook);
        // dataset[1].push_back(temp);
        // dataset[2].push_back(humidity);
        // dataset[3].push_back(wind);
        // dataset[4].push_back(play);
        //
        // ++rows;
    }

    Dataset dataset(row_data, target_data);

    const auto tree = build_tree(dataset);

    print_tree(tree);

    int correct = 0;
    for (int i = 0; i < dataset.num_rows(); ++i)
    {
        if (tree_predict(dataset.row_data[i], tree) == dataset.target_data[i])
        {
            ++correct;
        }
    }

    // int correct = 0;
    // for (int i = 0; i < rows; ++i)
    // {
    //     std::vector<int> obs;
    //     for (int attribute = 0; attribute < cols - 1; ++attribute)
    //     {
    //         obs.push_back(dataset[attribute][i]);
    //     }
    //     auto pred = tree_predict(obs, tree);
    //     if (pred == dataset.back()[i])
    //         ++correct;
    // }

    std::cout << "accuracy: " << static_cast<float>(correct) / dataset.num_rows() << '\n';

    return 0;
}

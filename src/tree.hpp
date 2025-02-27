#pragma once

#include <fmt/core.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Dataset = std::vector<std::vector<int>>;

inline std::pair<int, int> mode(const std::vector<int> &data)
{
    std::unordered_map<int, int> counts;

    int highest_count = 0;
    int ret = 0;

    for (auto x : data)
    {
        ++counts[x];
        if (counts[x] > highest_count)
        {
            highest_count = counts[x];
            ret = x;
        }
    }

    return {ret, highest_count};
}

inline float compute_entropy(const std::vector<int> &data)
{
    std::unordered_map<int, int> counts;

    for (auto x : data)
    {
        ++counts[x];
    }

    float ret = 0;
    for (auto [_, cnt] : counts)
    {
        float prop = static_cast<float>(cnt) / data.size();
        ret += -prop * log(prop);
    }

    return ret;
}

inline std::unordered_map<int, Dataset> split_dataset(const Dataset &dataset, int attribute)
{
    int rows = dataset[0].size();

    std::unordered_map<int, Dataset> label_splits;
    for (int row = 0; row < rows; ++row)
    {
        auto it = label_splits.find(dataset[attribute][row]);
        if (it == label_splits.end())
        {
            it = label_splits.insert({dataset[attribute][row], Dataset(dataset.size())}).first;
        }

        for (int col = 0; col < dataset.size(); ++col)
        {
            it->second[col].push_back(dataset[col][row]);
        }
    }

    return label_splits;
}

inline float split_entropy(const Dataset &dataset, int attribute)
{
    int rows = dataset[0].size();

    auto label_splits = split_dataset(dataset, attribute);

    float split_entropy = 0;

    for (const auto &[label, ds] : label_splits)
    {
        float entropy = compute_entropy(ds[4]);

        float prop = static_cast<float>(ds[0].size()) / rows;
        split_entropy += prop * entropy;
    }

    return split_entropy;
}

struct InterNode;
struct LeafNode;

using Node = std::variant<InterNode, LeafNode>;

struct InterNode
{
    int attribute;

    std::unordered_map<int, Node> children;
};

struct LeafNode
{
    int label;
};

inline void print_tree(const Node &node, int depth = 0)
{
    auto indent = std::string(depth * 2, ' ');
    std::visit(
        [&](const auto &n)
        {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, InterNode>)
            {
                std::cout << indent << "[Node: attribute=" << n.attribute << "]\n";
                for (const auto &[_, child] : n.children)
                {
                    print_tree(child, depth + 1);
                }
            }
            else
            {
                std::cout << indent << "[Leaf: label=" << n.label << "]\n";
            }
        },
        node);
}

inline int tree_predict(const std::vector<int> &obs, const Node &node)
{
    return std::visit(
        [&](const auto &n)
        {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, InterNode>)
            {
                return tree_predict(obs, n.children.at(obs[n.attribute]));
            }
            else
            {
                return n.label;
            }
        },
        node);
}

inline Node id3(const Dataset &dataset, std::unordered_set<int> used_attributes, int parent_mode, int min_samples_split)
{
    const int rows = dataset[0].size();
    const int num_attributes = dataset.size() - 1;
    const auto &target_data = dataset.back();

    if (rows == 0)
    {
        return LeafNode{parent_mode};
    }

    auto [mode_label, mode_count] = mode(target_data);

    if (mode_count == rows || used_attributes.size() == num_attributes || rows <= min_samples_split)
    {
        return LeafNode{mode_label};
    }

    float best_split_entropy = std::numeric_limits<float>::max();
    int best_split_attribute = 0;

    for (int col = 0; col < num_attributes; ++col)
    {
        if (used_attributes.contains(col))
            continue;

        auto entropy = split_entropy(dataset, col);
        if (entropy < best_split_entropy)
        {
            best_split_entropy = entropy;
            best_split_attribute = col;
        }
    }

    used_attributes.insert(best_split_attribute);

    const auto label_splits = split_dataset(dataset, best_split_attribute);

    std::unordered_map<int, Node> children;
    for (const auto &[label, split_ds] : label_splits)
    {
        children[label] = id3(split_ds, used_attributes, mode_label, min_samples_split);
    }

    return InterNode{best_split_attribute, std::move(children)};
}

inline Node build_tree(const Dataset &dataset, int min_samples_split = 2)
{
    auto [mode_label, _] = mode(dataset.back());

    return id3(dataset, std::unordered_set<int>{}, mode_label, min_samples_split);
}

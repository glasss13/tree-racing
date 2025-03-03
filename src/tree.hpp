#pragma once

#include "dataset.hpp"
#include <algorithm>
#include <bitset>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <random>
#include <vector>

std::pair<std::pair<std::vector<std::vector<int>>, std::vector<int>>, std::pair<std::vector<std::vector<int>>, std::vector<int>>> inline train_test_split(
    const std::vector<std::vector<int>> &row_data, const std::vector<int> &target_data, double train_ratio)
{
    size_t data_size = row_data.size();
    std::vector<size_t> indices(data_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    size_t train_size = static_cast<size_t>(train_ratio * data_size);

    std::vector<std::vector<int>> train_data, test_data;
    std::vector<int> train_target, test_target;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (i < train_size)
        {
            train_data.push_back(row_data[indices[i]]);
            train_target.push_back(target_data[indices[i]]);
        }
        else
        {
            test_data.push_back(row_data[indices[i]]);
            test_target.push_back(target_data[indices[i]]);
        }
    }

    return {{train_data, train_target}, {test_data, test_target}};
}

class Node
{
    int m_label_or_attr;
    std::vector<Node> m_children;
    int m_inter_label;

    Node(int label) : m_label_or_attr(label) {}

    Node(int attr, std::vector<Node> children, int inter_label) : m_label_or_attr(attr), m_children(std::move(children)), m_inter_label(inter_label) {}

  public:
    bool is_leaf() const { return m_children.empty(); }

    std::vector<Node> &children() { return m_children; }

    std::vector<Node> const &children() const { return m_children; }

    int split_attribute() const { return m_label_or_attr; }

    int leaf_label() const { return m_label_or_attr; }

    int inter_label() const { return m_inter_label; }

    void set_inter_label(int l) { m_inter_label = l; }

    static Node make_leaf(int label) { return Node{label}; }

    static Node make_inter(int attribute, std::vector<Node> children) { return Node{attribute, std::move(children), -1}; }
};

inline void print_tree(const Node &node, int depth = 0)
{
    auto indent = std::string(depth * 2, ' ');

    if (node.is_leaf())
    {
        std::cout << indent << "[Leaf: label=" << node.leaf_label() << "]\n";
    }
    else
    {
        std::cout << indent << "[Node: attribte=" << node.split_attribute() << "]\n";
        for (const auto &child : node.children())
        {
            print_tree(child, depth + 1);
        }
    }
}

inline int tree_predict(const std::vector<int> &obs, const Node &node)
{
    if (node.is_leaf())
    {
        return node.leaf_label();
    }
    else
    {
        int split_val = obs[node.split_attribute()];
        auto child = std::ranges::find_if(node.children(), [&](const Node &n) { return n.inter_label() == split_val; });
        if (child == node.children().end()) [[unlikely]]
        {
            return tree_predict(obs, node.children().front());
        }

        return tree_predict(obs, *child);
    }
}

inline float split_entropy(Dataset &ds, int attribute)
{
    float total_entropy = 0;

    const auto compute_entropy = [](int total, const std::vector<int> &counts)
    {
        float entropy = 0;
        for (auto cnt : counts)
        {
            if (cnt == 0)
                continue;
            float prop = static_cast<float>(cnt) / total;
            entropy -= prop * log(prop);
        }
        return entropy * total;
    };

    auto &cnts = ds.count_scratch_buf();

    int split_start = 0;
    auto prev_label = ds.get_row_sorted(0)[attribute];
    int row = 0;
    while (row < ds.num_rows())
    {
        const auto cur_label = ds.get_col_sorted(attribute, row);

        if (cur_label == prev_label)
        {
            const auto label = ds.get_target_sorted(row);
            ++cnts[label];
            ++row;
        }
        else
        {
            total_entropy += compute_entropy(row - split_start, cnts);

            split_start = row;
            prev_label = cur_label;
            std::memset(cnts.data(), 0, cnts.size() * sizeof(int));
        }
    }

    if (row > split_start)
    {
        total_entropy += compute_entropy(row - split_start, cnts);
        std::memset(cnts.data(), 0, cnts.size() * sizeof(int));
    }

    return total_entropy;
}

inline Node id3(Dataset dataset, std::bitset<64> used_attributes, int parent_mode, int min_samples_split)
{
    if (dataset.num_rows() == 0)
    {
        return Node::make_leaf(parent_mode);
    }

    auto [mode_label, mode_count] = dataset.mode_label();

    if (mode_count == dataset.num_rows() || used_attributes.size() == dataset.num_attributes() || dataset.num_rows() <= min_samples_split)
    {
        return Node::make_leaf(mode_label);
    }

    float best_split_entropy = std::numeric_limits<float>::max();
    int best_split_attribute = 0;
    Dataset best_ds;

    for (int col = 0; col < dataset.num_attributes(); ++col)
    {
        if (used_attributes.test(col))
            continue;

        dataset.sort_by(col);

        const auto entropy = split_entropy(dataset, col);
        if (entropy < best_split_entropy)
        {
            best_split_entropy = entropy;
            best_split_attribute = col;
            best_ds = dataset;
        }
    }

    used_attributes.set(best_split_attribute);

    std::vector<Node> children;
    for (const auto split_ds : best_ds.split_iterator(best_split_attribute))
    {
        auto label = split_ds.get_row_sorted(0)[best_split_attribute];
        auto n = id3(split_ds, used_attributes, mode_label, min_samples_split);
        n.set_inter_label(label);
        children.push_back(std::move(n));
    }

    return Node::make_inter(best_split_attribute, std::move(children));
}

inline Node build_tree(std::vector<std::vector<int>> row_data, std::vector<int> target_data, int min_samples_split = 2)
{
    Dataset ds(std::make_shared<InnerDataset>(std::move(row_data), std::move(target_data)));

    return id3(ds, 0, 0, min_samples_split);
}

// inline Node build_tree(std::vector<std::vector<int>> row_data, std::vector<std::vector<int>> col_data, std::vector<int> target_data, int min_samples_split =
// 2)
// {
//     Dataset ds(std::make_shared<InnerDataset>(std::move(row_data), std::move(col_data), std::move(target_data)));
//
//     return id3(ds, 0, 0, min_samples_split);
// }

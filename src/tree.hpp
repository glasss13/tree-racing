#pragma once

#include "dataset.hpp"
#include <algorithm>
#include <fmt/core.h>
#include <iostream>
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

    const auto &attribute_col = dataset[attribute];
    const auto &target_col = dataset.back();

    std::unordered_map<int, std::unordered_map<int, int>> split_counts;
    std::unordered_map<int, int> split_totals;
    for (int i = 0; i < rows; ++i)
    {
        ++split_counts[attribute_col[i]][target_col[i]];
        ++split_totals[attribute_col[i]];
    }

    float split_entropy = 0;

    for (const auto &[split, counts] : split_counts)
    {
        auto tot = split_totals[split];
        float entropy = 0;
        for (auto [_, cnt] : counts)
        {
            float prop = static_cast<float>(cnt) / tot;
            entropy += -prop * log(prop);
        }

        float prop = static_cast<float>(tot) / rows;
        split_entropy += prop * entropy;
    }

    return split_entropy;
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

inline Node id3(const Dataset &dataset, std::bitset<64> used_attributes, int parent_mode, int min_samples_split)
{
    const int rows = dataset[0].size();
    const int num_attributes = dataset.size() - 1;
    const auto &target_data = dataset.back();

    if (rows == 0)
    {
        return Node::make_leaf(parent_mode);
    }

    auto [mode_label, mode_count] = mode(target_data);

    if (mode_count == rows || used_attributes.size() == num_attributes || rows <= min_samples_split)
    {
        return Node::make_leaf(mode_label);
    }

    float best_split_entropy = std::numeric_limits<float>::max();
    int best_split_attribute = 0;

    for (int col = 0; col < num_attributes; ++col)
    {
        if (used_attributes.test(col))
            continue;

        auto entropy = split_entropy(dataset, col);
        if (entropy < best_split_entropy)
        {
            best_split_entropy = entropy;
            best_split_attribute = col;
        }
    }

    used_attributes.set(best_split_attribute);

    const auto label_splits = split_dataset(dataset, best_split_attribute);

    std::vector<Node> children;
    for (const auto &[label, split_ds] : label_splits)
    {
        auto n = id3(split_ds, used_attributes, mode_label, min_samples_split);
        n.set_inter_label(label);
        children.push_back(std::move(n));
    }

    return Node::make_inter(best_split_attribute, std::move(children));
}

inline Node build_tree(const Dataset &dataset, int min_samples_split = 2)
{
    auto [mode_label, _] = mode(dataset.back());

    return id3(dataset, 0, mode_label, min_samples_split);
}

inline std::unordered_map<int, Dataset2> split_dataset2(const Dataset2 &ds, int attribute)
{
    std::unordered_map<int, Dataset2> label_splits;
    for (int row = 0; row < ds.num_rows(); ++row)
    {
        auto attr_label = ds.row_data[row][attribute];
        auto it = label_splits.find(attr_label);
        if (it == label_splits.end())
        {
            it = label_splits.insert({attr_label, Dataset2()}).first;
        }

        auto &split_ds = it->second;
        split_ds.row_data.push_back(ds.row_data[row]);
        split_ds.target_data.push_back(ds.target_data[row]);
    }

    return label_splits;
}

inline float split_entropy2(const Dataset2 &ds, int attribute)
{
    std::unordered_map<int, std::unordered_map<int, int>> split_counts;
    std::unordered_map<int, int> split_totals;
    for (int i = 0; i < ds.num_rows(); ++i)
    {
        ++split_counts[ds.row_data[i][attribute]][ds.target_data[i]];
        ++split_totals[ds.row_data[i][attribute]];
    }

    float split_entropy = 0;

    for (const auto &[split, counts] : split_counts)
    {
        auto tot = split_totals[split];
        float entropy = 0;
        for (auto [_, cnt] : counts)
        {
            float prop = static_cast<float>(cnt) / tot;
            entropy += -prop * log(prop);
        }

        float prop = static_cast<float>(tot) / ds.num_rows();
        split_entropy += prop * entropy;
    }

    return split_entropy;
}

inline Node id32(const Dataset2 &dataset, std::bitset<64> used_attributes, int parent_mode, int min_samples_split)
{
    if (dataset.num_rows() == 0)
    {
        return Node::make_leaf(parent_mode);
    }

    auto [mode_label, mode_count] = mode(dataset.target_data);

    if (mode_count == dataset.num_rows() || used_attributes.size() == dataset.num_attributes() || dataset.num_rows() <= min_samples_split)
    {
        return Node::make_leaf(mode_label);
    }

    float best_split_entropy = std::numeric_limits<float>::max();
    int best_split_attribute = 0;

    for (int col = 0; col < dataset.num_attributes(); ++col)
    {
        if (used_attributes.test(col))
            continue;

        auto entropy = split_entropy2(dataset, col);
        if (entropy < best_split_entropy)
        {
            best_split_entropy = entropy;
            best_split_attribute = col;
        }
    }

    used_attributes.set(best_split_attribute);

    const auto label_splits = split_dataset2(dataset, best_split_attribute);

    std::vector<Node> children;
    for (const auto &[label, split_ds] : label_splits)
    {
        auto n = id32(split_ds, used_attributes, mode_label, min_samples_split);
        n.set_inter_label(label);
        children.push_back(std::move(n));
    }

    return Node::make_inter(best_split_attribute, std::move(children));
}

inline Node build_tree2(const Dataset2 &dataset, int min_samples_split = 2)
{
    auto [mode_label, _] = mode(dataset.target_data);

    return id32(dataset, 0, mode_label, min_samples_split);
}

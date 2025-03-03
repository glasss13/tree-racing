#pragma once

#include <vector>

using NodeHandle = std::size_t;

class Node
{
    int m_label_or_attr;
    std::vector<NodeHandle> m_children;
    int m_inter_label;

  public:
    Node(int label) : m_label_or_attr(label) {}

    Node(int attr, std::vector<NodeHandle> children, int inter_label) : m_label_or_attr(attr), m_children(std::move(children)), m_inter_label(inter_label) {}

    bool is_leaf() const { return m_children.empty(); }

    std::vector<NodeHandle> &children() { return m_children; }

    std::vector<NodeHandle> const &children() const { return m_children; }

    int split_attribute() const { return m_label_or_attr; }

    int leaf_label() const { return m_label_or_attr; }

    int inter_label() const { return m_inter_label; }

    void set_inter_label(int l) { m_inter_label = l; }
};

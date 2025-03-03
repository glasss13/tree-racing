#pragma once

#include <vector>

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

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

struct NodePool
{
    Node *buffer;
    std::size_t capacity;
    std::size_t next = 0;

    NodePool(std::size_t cap) : capacity(cap) { buffer = static_cast<Node *>(::operator new(capacity * sizeof(Node))); }

    ~NodePool() { ::operator delete(buffer); }

    NodeHandle make_leaf(int label)
    {
        new (&buffer[next]) Node(label);
        return next++;
    }

    NodeHandle make_inter(int attribute, std::vector<NodeHandle> children)
    {
        new (&buffer[next]) Node(attribute, std::move(children), -1);
        return next++;
    }

    Node &get(NodeHandle handle) { return buffer[handle]; }

    const Node &get(NodeHandle handle) const { return buffer[handle]; }
};

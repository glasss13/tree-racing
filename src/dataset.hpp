#pragma once

#include <algorithm>
#include <fmt/core.h>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>

struct InnerDataset
{
    std::vector<std::vector<int>> m_row_data;
    std::vector<std::vector<int>> m_col_data;
    std::vector<int> m_target_data;
    std::vector<int> m_count_scratch_buf;

    InnerDataset(std::vector<std::vector<int>> row_data, std::vector<int> target) : m_row_data(std::move(row_data)), m_target_data(std::move(target))
    {
        m_col_data.resize(m_row_data[0].size(), std::vector<int>(m_row_data.size()));

        int mx = 0;
        for (auto t : m_target_data)
        {
            mx = std::max(mx, t);
        }
        for (const auto &r : m_row_data)
        {
            for (auto v : r)
            {
                mx = std::max(mx, v);
            }
        }
        m_count_scratch_buf.resize(mx + 1, 0);
        for (size_t i = 0; i < m_row_data.size(); ++i)
        {
            for (size_t j = 0; j < m_row_data[0].size(); ++j)
            {
                m_col_data[j][i] = m_row_data[i][j];
            }
        }
    }
};

class Dataset
{
    InnerDataset *m_inner;
    std::vector<int> m_sorted_idxs;

  public:
    explicit Dataset(InnerDataset *inner) : m_inner(std::move(inner)), m_sorted_idxs(m_inner->m_row_data.size())
    {
        std::iota(m_sorted_idxs.begin(), m_sorted_idxs.end(), 0);
    }

    Dataset(InnerDataset *inner, size_t nrows) : m_inner(inner), m_sorted_idxs(nrows) {}

    Dataset() = default;

    void sort_by(size_t col)
    {
        const auto &col_data = m_inner->m_col_data[col];
        size_t n = m_sorted_idxs.size();
        auto &count = m_inner->m_count_scratch_buf;

        int max_val = 0;
        for (int idx : m_sorted_idxs)
        {
            int key = col_data[idx];
            ++count[key];
            max_val = std::max(max_val, key);
        }

        for (size_t i = 1; i <= max_val; i++)
        {
            count[i] += count[i - 1];
        }

        std::vector<int> output(n);
        for (int i = n - 1; i >= 0; i--)
        {
            int idx = m_sorted_idxs[i];
            int key = col_data[idx];
            int pos = count[key] - 1;
            output[pos] = idx;
            count[key]--;
        }

        m_sorted_idxs = std::move(output);
        std::memset(count.data(), 0, count.size() * sizeof(int));
    }

    std::vector<int> &count_scratch_buf() { return m_inner->m_count_scratch_buf; }

    const std::vector<int> &get_row(int row) const { return m_inner->m_row_data[row]; }

    int get_target(int row) const { return m_inner->m_target_data[row]; }

    const std::vector<int> &get_row_sorted(int row) const { return m_inner->m_row_data[m_sorted_idxs[row]]; }

    int get_col_sorted(int col, int entry) const { return m_inner->m_col_data[col][m_sorted_idxs[entry]]; }

    int get_target_sorted(int row) const { return m_inner->m_target_data[m_sorted_idxs[row]]; }

    size_t num_rows() const { return m_sorted_idxs.size(); }

    size_t num_attributes() const { return m_inner->m_col_data.size(); }

    const std::vector<int> &get_target_data() const { return m_inner->m_target_data; }

    std::pair<int, int> mode_label() const
    {
        auto &counts = m_inner->m_count_scratch_buf;

        int highest_count = 0;
        int ret = 0;

        for (auto idx : m_sorted_idxs)
        {
            const auto x = m_inner->m_target_data[idx];

            ++counts[x];
            if (counts[x] > highest_count)
            {
                highest_count = counts[x];
                ret = x;
            }
        }

        std::memset(counts.data(), 0, counts.size() * sizeof(int));

        return {ret, highest_count};
    }

    Dataset copy_from(size_t start, size_t end) const
    {
        const auto count = end - start;

        Dataset ret(m_inner, count);
        std::memcpy(ret.m_sorted_idxs.data(), m_sorted_idxs.data() + start, count * sizeof(int));

        return ret;
    }

    size_t find_next_label(int col, int label) const
    {
        const auto &col_data = m_inner->m_col_data[col];
        auto it = std::ranges::upper_bound(m_sorted_idxs, label, std::less<>(), [&](int idx) { return col_data[idx]; });
        return std::distance(m_sorted_idxs.begin(), it);
    }

  private:
    class SplitDatasetIterator
    {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Dataset;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = Dataset;

        SplitDatasetIterator(const Dataset *ds, int attribute, size_t idx) : m_ds(ds), m_attribute(attribute), m_idx(idx) {}

        Dataset operator*() const
        {
            size_t next_idx = m_ds->find_next_label(m_attribute, m_ds->get_col_sorted(m_attribute, m_idx));
            return m_ds->copy_from(m_idx, next_idx);
        }

        SplitDatasetIterator &operator++()
        {
            size_t next_idx = m_ds->find_next_label(m_attribute, m_ds->get_col_sorted(m_attribute, m_idx));
            m_idx = next_idx;
            return *this;
        }

        SplitDatasetIterator operator++(int)
        {
            SplitDatasetIterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const SplitDatasetIterator &other) const { return m_ds == other.m_ds && m_idx == other.m_idx && m_attribute == other.m_attribute; }

        bool operator!=(const SplitDatasetIterator &other) const { return !(*this == other); }

      private:
        const Dataset *m_ds;
        int m_attribute;
        size_t m_idx;
    };

    class SplitDatasetView
    {
      public:
        SplitDatasetView(const Dataset &ds, int attribute) : m_ds(ds), m_attribute(attribute) {}

        SplitDatasetIterator begin() const { return SplitDatasetIterator(&m_ds, m_attribute, 0); }

        SplitDatasetIterator end() const { return SplitDatasetIterator(&m_ds, m_attribute, m_ds.num_rows()); }

      private:
        const Dataset &m_ds;
        int m_attribute;
    };

  public:
    SplitDatasetView split_iterator(int attribute) const { return SplitDatasetView(*this, attribute); }
};

#pragma once

#include <algorithm>
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

    InnerDataset(std::vector<std::vector<int>> row_data, std::vector<int> target)
        : m_row_data(std::move(row_data)), m_target_data(std::move(target)), m_count_scratch_buf(32)
    {
        m_col_data.resize(m_row_data[0].size(), std::vector<int>(m_row_data.size()));

        for (size_t i = 0; i < m_row_data.size(); ++i)
        {
            for (size_t j = 0; j < m_row_data[0].size(); ++j)
            {
                m_col_data[j][i] = m_row_data[i][j];
            }
        }
    }

    InnerDataset(std::vector<std::vector<int>> row_data, std::vector<std::vector<int>> col_data, std::vector<int> target)
        : m_row_data(std::move(row_data)), m_col_data(std::move(col_data)), m_target_data(std::move(target)), m_count_scratch_buf(32)
    {
    }
};

class Dataset
{
    std::shared_ptr<InnerDataset> m_inner;
    std::vector<int> m_sorted_idxs;

    // public:
    //   static int cnt;
    //
    //   static int get_cnt() { return cnt; }

  public:
    explicit Dataset(std::shared_ptr<InnerDataset> inner) : m_inner(std::move(inner)), m_sorted_idxs(m_inner->m_row_data.size())
    {
        // ++cnt;
        std::iota(m_sorted_idxs.begin(), m_sorted_idxs.end(), 0);
    }

    Dataset(std::shared_ptr<InnerDataset> inner, int nrows) : m_inner(inner), m_sorted_idxs(nrows) {}

    Dataset() = default;

    void sort_by(int col)
    {
        const auto &col_data = m_inner->m_col_data[col];
        std::ranges::sort(m_sorted_idxs, [&](int l, int r) { return col_data[l] < col_data[r]; });
    }

    void sort_by_target()
    {
        const auto &target_data = m_inner->m_target_data;
        std::ranges::sort(m_sorted_idxs, [&](int l, int r) { return target_data[l] < target_data[r]; });
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

    // TODO: clean
    std::pair<int, int> mode_label() const
    {
        auto &counts = m_inner->m_count_scratch_buf;

        int highest_count = 0;
        int ret = 0;

        for (auto idx : m_sorted_idxs)
        {
            auto x = m_inner->m_target_data[idx];
            ++counts[x];
            if (counts[x] > highest_count)
            {
                highest_count = counts[x];
                ret = x;
            }
        }

        std::memset(counts.data(), 0, counts.size());

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

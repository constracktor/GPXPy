#include "../include/automobile_bits/utils_c.hpp"

// #include <iostream>
// #include <hpx/hpx.hpp>
// #include <hpx/future.hpp>
// #include <hpx/iostream.hpp>

// #include <hpx/hpx_start.hpp>
// #include <hpx/include/post.hpp>

namespace utils
{
    int compute_train_tiles(int n_samples, int n_tile_size)
    {
        std::size_t _n_samples = static_cast<std::size_t>(n_samples);
        std::size_t _n_tile_size = static_cast<std::size_t>(n_tile_size);
        std::size_t _n_tiles = _n_samples / _n_tile_size;
        return static_cast<int>(_n_tiles);
    }

    std::pair<int, int> compute_test_tiles(int n_test, int n_tiles, int n_tile_size)
    {
        std::size_t _n_test = static_cast<std::size_t>(n_test);
        std::size_t _n_tiles = static_cast<std::size_t>(n_tiles);
        std::size_t _n_tile_size = static_cast<std::size_t>(n_tile_size);
        std::size_t m_tiles;
        std::size_t m_tile_size;

        if ((_n_test % _n_tile_size) > 0)
        {
            m_tiles = _n_tiles;
            m_tile_size = _n_test / m_tiles;
        }
        else
        {
            m_tiles = _n_test / _n_tile_size;
            m_tile_size = _n_tile_size;
        }
        return {static_cast<int>(m_tiles), static_cast<int>(m_tile_size)};
    }
}

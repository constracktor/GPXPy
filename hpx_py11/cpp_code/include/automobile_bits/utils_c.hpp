#include <stdio.h>
#include <tuple>
// #include <iostream>
// #include <hpx/hpx_init.hpp>
// #include <hpx/future.hpp>

#ifndef UTILS_C_H
#define UTILS_C_H

namespace utils
{

    /// Function to load data from a file into data
    int compute_train_tiles(int n_samples, int n_tile_size);
    std::pair<int, int> compute_test_tiles(int m_samples, int n_tiles, int n_tile_size);

    // void print(const std::vector<double> &vec, int start = 0, int end = -1, const std::string &separator = " ");

}

#endif

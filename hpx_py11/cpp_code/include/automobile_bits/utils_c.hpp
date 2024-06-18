#include <stdio.h>
#include <tuple>
#include <stdexcept>
#include <vector>
#include <string>

#include <hpx/hpx_start.hpp>
#include <hpx/include/post.hpp>

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
    
    /// Function to load data from a file
    std::vector<double> load_data(const std::string &file_path, int n_samples);

    void print(const std::vector<double> &vec, int start = 0, int end = -1, const std::string &separator = " ");

    void start_hpx_runtime(int argc, char** argv);

    void stop_hpx_runtime();

}

#endif

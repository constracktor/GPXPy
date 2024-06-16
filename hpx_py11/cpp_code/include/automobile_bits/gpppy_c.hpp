#include <vector>
#include <string>
// #include <iostream>

// #include <hpx/hpx_init.hpp>
// #include <hpx/future.hpp>

#ifndef GPPPY_C_H
#define GPPPY_C_H

namespace gpppy
{

    struct GP_data
    {
        int n_samples;
        std::string file_path;
        std::vector<double> data;

        GP_data(const std::string &f_path, int n);
    };

    /// Function to load data from a file into data
    std::vector<double> load_data(const std::string &file_path, int n_samples);

    class GP
    {

    private:
        std::vector<double> training_input;
        // std::vector<double> training_output;
        // std::vector<double> test_input;
        // std::vector<double> test_output;

        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> K_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_v_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_l_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> grad_K_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prior_K_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> alpha_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> y_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> cross_covariance_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> t_cross_covariance_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_tiles;
        // std::vector<hpx::shared_future<std::vector<CALC_TYPE>>> prediction_uncertainty_tiles;

        // public:
        //     /// Constructor
        //     GP();

        //     void start_hpx_runtime(int argc, char** argv);

        //     void stop_hpx_runtime();

        //     void set(int s);

        //     /// Get name
        //     /// @return Name
        //     std::string get_stud_id() const;

        //     void do_fut() const;

        //     int add(int i, int j);
    };

}

#endif

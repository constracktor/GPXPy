#ifndef GPPPY_C_H
#define GPPPY_C_H

#include <vector>
#include <string>

// #include <hpx/hpx_init.hpp>

namespace gpppy
{

    struct GP_data
    {
        int n_samples;
        std::string file_path;
        std::vector<double> data;

        GP_data(const std::string &f_path, int n);
    };

    struct Kernel_Params
    {
        double lengthscale;
        double vertical_lengthscale;
        double noise_variance;
        int n_regressors;

        Kernel_Params(double l = 1.0, double v = 1.0, double n = 0.1, int n_r = 100);

        std::string repr() const;
    };

    struct Hyperparameters
    {
        double learning_rate;
        double beta1;
        double beta2;
        double epsilon;
        int opt_iter;

        Hyperparameters(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, int opt_i = 0);

        std::string repr() const;
    };

    class GP
    {

    private:
        std::vector<double> _training_input;
        std::vector<double> _training_output;
        int _n_tiles;
        int _n_tile_size;

    public:
        double lengthscale;
        double vertical_lengthscale;
        double noise_variance;
        int n_regressors;

        /// Constructor
        GP(std::vector<double> input, std::vector<double> output, int n_tiles, int n_tile_size, double l, double v, double n, int n_r);

        std::string repr() const;

        std::vector<double> get_training_input() const;

        std::vector<double> get_training_output() const;

        std::vector<std::vector<double>> predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

        //     void set(int s);

        //     /// Get name
        //     /// @return Name
        //     std::string get_stud_id() const;

        //     void do_fut() const;

        //     int add(int i, int j);
    };

}

#endif

#include "../core/include/gp_functions.hpp"
#include "../core/include/gpxpy_c.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Adds classes `GP_data`, `Hyperparameters`, `GP` to Python module.
 */
void init_gpxpy(py::module& m)
{
    // set training data with `GP_data` class
    py::class_<gpxpy::GP_data>(
        m, "GP_data", "Class representing Gaussian Process data.")
        .def(py::init<std::string, int>(),
             py::arg("file_path"),
             py::arg("n_samples"),
             R"pbdoc(
             Loads data for Gaussian Process from file.

             Parameters:
                 file_path (str): Path to the file containing the GP data.
                 n_samples (int): Number of samples in the GP data.
             )pbdoc")
        .def_readonly("n_samples",
                      &gpxpy::GP_data::n_samples,
                      "Number of samples in the GP data")
        .def_readonly(
            "file_path", &gpxpy::GP_data::file_path, "File path to the GP data")
        .def_readonly(
            "data", &gpxpy::GP_data::data, "Data in the GP data file");

    // Set hyperparameters to default values in `Hyperparameters` class, unless
    // specified. Python object has full access to each hyperparameter and a
    // string representation `__repr__`.
    py::class_<gpxpy_hyper::Hyperparameters>(m, "Hyperparameters")
        .def(py::init<double,
                      double,
                      double,
                      double,
                      int,
                      std::vector<double>,
                      std::vector<double>>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("opt_iter") = 0,
             py::arg("m_T") = std::vector<double>{0.0, 0.0, 0.0},
             py::arg("v_T") = std::vector<double>{0.0, 0.0, 0.0})
        .def_readwrite("learning_rate",
                       &gpxpy_hyper::Hyperparameters::learning_rate)
        .def_readwrite("beta1", &gpxpy_hyper::Hyperparameters::beta1)
        .def_readwrite("beta2", &gpxpy_hyper::Hyperparameters::beta2)
        .def_readwrite("epsilon", &gpxpy_hyper::Hyperparameters::epsilon)
        .def_readwrite("opt_iter", &gpxpy_hyper::Hyperparameters::opt_iter)
        .def_readwrite("m_T", &gpxpy_hyper::Hyperparameters::M_T)
        .def_readwrite("v_T", &gpxpy_hyper::Hyperparameters::V_T)
        .def("__repr__", &gpxpy_hyper::Hyperparameters::repr);
    ;

    // Initializes Gaussian Process with `GP` class. Sets default parameters for
    // squared exponential kernel, number of regressors and trainable, unless
    // specified. Instance object has full access to parameters for squared
    // exponential kernel and number of regressors. Also adds some member
    // functions.
    // GPU support is disabled by default and may only be enabled on
    // initialization.
    py::class_<gpxpy::GP>(m, "GP")
        .def(py::init<std::vector<double>,
                      std::vector<double>,
                      int,
                      int,
                      double,
                      double,
                      double,
                      int,
                      std::vector<bool>>(),
             py::arg("input_data"),
             py::arg("output_data"),
             py::arg("n_tiles"),
             py::arg("n_tile_size"),
             py::arg("lengthscale") = 1.0,
             py::arg("v_lengthscale") = 1.0,
             py::arg("noise_var") = 0.1,
             py::arg("n_reg") = 100,
             py::arg("trainable") = std::vector<bool>{true, true, true},
             R"pbdoc(
Create Gaussian Process including its data, hyperparameters.

Parameters:
    input_data (list): Input data for the GP.
    output_data (list): Output data for the GP.
    n_tiles (int): Number of tiles to split the input data.
    n_tile_size (int): Size of each tile.
    lengthscale (float): Lengthscale hyperparameter for the squared exponential
        kernel. Default is 1.
    v_lengthscale (float): Vertical lengthscale for the squared exponential
        kernel. Default is 1.
    noise_var (float): Noise variance for the squared exponential kernel.
        Default is 0.1.
    n_reg (int): Number of regressors. Default is 100.
    trainable (list): List of booleans for trainable hyperparameters. Default is
        {true, true, true}.
             )pbdoc")
        .def_readwrite("lengthscale", &gpxpy::GP::lengthscale)
        .def_readwrite("v_lengthscale", &gpxpy::GP::vertical_lengthscale)
        .def_readwrite("noise_var", &gpxpy::GP::noise_variance)
        .def_readwrite("n_reg", &gpxpy::GP::n_regressors)
        .def("__repr__", &gpxpy::GP::repr)
        .def("get_input_data", &gpxpy::GP::get_training_input)
        .def("get_output_data", &gpxpy::GP::get_training_output)
        .def("predict",
             &gpxpy::GP::predict,
             py::arg("test_data"),
             py::arg("m_tiles"),
             py::arg("m_tile_size"))
        .def("predict_with_uncertainty",
             &gpxpy::GP::predict_with_uncertainty,
             py::arg("test_data"),
             py::arg("m_tiles"),
             py::arg("m_tile_size"))
        .def("predict_with_full_cov",
             &gpxpy::GP::predict_with_full_cov,
             py::arg("test_data"),
             py::arg("m_tiles"),
             py::arg("m_tile_size"))
        .def("optimize", &gpxpy::GP::optimize, py::arg("hyperparams"))
        .def("optimize_step",
             &gpxpy::GP::optimize_step,
             py::arg("hyperparams"),
             py::arg("iter"))
        .def("compute_loss", &gpxpy::GP::calculate_loss);
}

#include "../cpp_code/include/gp_headers/gpppy_c.hpp"
#include "../cpp_code/include/gp_headers/gp_functions.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_gpppy(py::module &m)
{
    py::class_<gpppy::GP_data>(m, "GP_data")
        .def(py::init<std::string, int>(), py::arg("file_path"), py::arg("n_samples"))
        .def_readonly("n_samples", &gpppy::GP_data::n_samples)
        .def_readonly("file_path", &gpppy::GP_data::file_path)
        .def_readonly("data", &gpppy::GP_data::data);

    py::class_<gpppy_hyper::Hyperparameters>(m, "Hyperparameters")
        .def(py::init<double, double, double, double, int, std::vector<double>, std::vector<double>>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("opt_iter") = 0,
             py::arg("m_T") = std::vector<double>{0.0, 0.0, 0.0},
             py::arg("v_T") = std::vector<double>{0.0, 0.0, 0.0})
        .def_readwrite("learning_rate", &gpppy_hyper::Hyperparameters::learning_rate)
        .def_readwrite("beta1", &gpppy_hyper::Hyperparameters::beta1)
        .def_readwrite("beta2", &gpppy_hyper::Hyperparameters::beta2)
        .def_readwrite("epsilon", &gpppy_hyper::Hyperparameters::epsilon)
        .def_readwrite("opt_iter", &gpppy_hyper::Hyperparameters::opt_iter)
        .def_readwrite("m_T", &gpppy_hyper::Hyperparameters::M_T)
        .def_readwrite("v_T", &gpppy_hyper::Hyperparameters::V_T)
        .def("__repr__", &gpppy_hyper::Hyperparameters::repr);
    ;

    py::class_<gpppy::GP>(m, "GP")
        .def(py::init<std::vector<double>, std::vector<double>, int, int, double, double, double, int, std::vector<bool>>(),
             py::arg("input_data"), py::arg("output_data"),
             py::arg("n_tiles"), py::arg("n_tile_size"),
             py::arg("lengthscale") = 1.0,
             py::arg("v_lengthscale") = 1.0,
             py::arg("noise_var") = 0.1,
             py::arg("n_reg") = 100,
             py::arg("trainable") = std::vector<bool>{true, true, true})
        .def_readwrite("lengthscale", &gpppy::GP::lengthscale)
        .def_readwrite("v_lengthscale", &gpppy::GP::vertical_lengthscale)
        .def_readwrite("noise_var", &gpppy::GP::noise_variance)
        .def_readwrite("n_reg", &gpppy::GP::n_regressors)
        .def("__repr__", &gpppy::GP::repr)
        .def("get_input_data", &gpppy::GP::get_training_input)
        .def("get_output_data", &gpppy::GP::get_training_output)
        .def("predict", &gpppy::GP::predict, py::arg("test_data"), py::arg("m_tiles"), py::arg("m_tile_size"))
        .def("optimize", &gpppy::GP::optimize, py::arg("hyperparams"))
        .def("optimize_step", &gpppy::GP::optimize_step, py::arg("hyperparams"), py::arg("iter"))
        .def("compute_loss", &gpppy::GP::calculate_loss);
}
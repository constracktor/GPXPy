#include "../cpp_code/include/automobile_bits/gpppy_c.hpp"
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

    py::class_<gpppy::Kernel_Params>(m, "Kernel_Params")
        .def(py::init<double, double, double, int>(),
             py::arg("lengthscale") = 1.0,
             py::arg("v_lengthscale") = 1.0,
             py::arg("noise_var") = 0.1,
             py::arg("n_reg") = 100)
        .def_readwrite("lengthscale", &gpppy::Kernel_Params::lengthscale)
        .def_readwrite("v_lengthscale", &gpppy::Kernel_Params::vertical_lengthscale)
        .def_readwrite("noise_var", &gpppy::Kernel_Params::noise_variance)
        .def_readwrite("n_reg", &gpppy::Kernel_Params::n_regressors)
        .def("__repr__", &gpppy::Kernel_Params::repr);

    py::class_<gpppy::Hyperparameters>(m, "Hyperparameters")
        .def(py::init<double, double, double, double, int>(),
             py::arg("learning_rate") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 1e-8,
             py::arg("opt_iter") = 0)
        .def_readwrite("learning_rate", &gpppy::Hyperparameters::learning_rate)
        .def_readwrite("beta1", &gpppy::Hyperparameters::beta1)
        .def_readwrite("beta2", &gpppy::Hyperparameters::beta2)
        .def_readwrite("epsilon", &gpppy::Hyperparameters::epsilon)
        .def_readwrite("opt_iter", &gpppy::Hyperparameters::opt_iter)
        .def("__repr__", &gpppy::Hyperparameters::repr);
    ;

    py::class_<gpppy::GP>(m, "GP")
        .def(py::init<std::vector<double>, std::vector<double>>(), py::arg("input_data"), py::arg("output_data"))
        .def("get_input_data", &gpppy::GP::get_training_input)
        .def("get_output_data", &gpppy::GP::get_training_output);
}
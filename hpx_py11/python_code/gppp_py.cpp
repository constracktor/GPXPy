#include "../cpp_code/include/automobile_bits/gpppy_c.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_gpppy(py::module &m)
{
    py::class_<gpppy::GP_data>(m, "GP_data")
        .def(py::init<std::string, int>(), py::arg("file_path"), py::arg("n_samples"))
        .def_readonly("data", &gpppy::GP_data::data)
        .def_readonly("file_path", &gpppy::GP_data::file_path)
        .def_readonly("n_samples", &gpppy::GP_data::n_samples);
}

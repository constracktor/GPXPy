#include "../cpp_code/include/automobile_bits/motorcycle.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_motorcycle(py::module &m)
{
    py::class_<vehicles::Motorcycle>(m, "Motorcycle")
        .def(py::init<std::string>(), py::arg("name"))
        .def("get_name", &vehicles::Motorcycle::get_name)
        .def("ride", &vehicles::Motorcycle::ride, py::arg("road"));
}

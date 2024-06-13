#include "../cpp_code/include/automobile_bits/motorcycle.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


void start_hpx_wrapper(university::Student& student, std::vector<std::string> args)
{
    std::vector<char*> argv;
    for (auto& arg : args)
        argv.push_back(&arg[0]);
    argv.push_back(nullptr);
    int argc = args.size();
    student.start_hpx(argc, argv.data());
}

void init_universtiy(py::module &m)
{
    py::class_<university::Student>(m, "Student")
        .def(py::init<std::string>(), py::arg("stud_id"))
        .def("get_stud_id", &university::Student::get_stud_id)
        .def("do_fut", &university::Student::do_fut)
        .def("add", &university::Student::add, py::arg("i"), py::arg("j"))
        // .def("start_hpx",&university::Student::start_hpx, py::arg("argc"), py::arg("argv"))
        .def("start_hpx", &start_hpx_wrapper, py::arg("args")) // Using the wrapper function
        .def("stop_hpx",&university::Student::stop_hpx);
}

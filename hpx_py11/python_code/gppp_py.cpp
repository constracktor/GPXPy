#include "../cpp_code/include/automobile_bits/gpppy_c.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// void print_slice(const std::vector<double>& vec, py::slice slice, const std::string& separator = " ") {
    
//     py::str slice_str = py::str(slice);
//     std::cout << "Slice: " << slice_str << std::endl;
//     std::cout << "Separator: " << separator << std::endl;

//     size_t start, stop, step, slicelength;
//     if (!slice.compute(vec.size(), &start, &stop, &step, &slicelength)) {
//         throw py::error_already_set();
//     }

//     std::cout << "start: " << start << std::endl;
//     std::cout << "stop: " << stop << std::endl;
//     std::cout << "step: " << step << std::endl;
//     std::cout << "slicelength: " << slicelength << std::endl;

//     // Call the original print function with sliced vector
//     gpppy::print(vec, start, stop, separator);
// }


void init_gpppy(py::module &m)
{
    py::class_<gpppy::GP_data>(m, "GP_data")
        .def(py::init<std::string, int>(), py::arg("file_path"), py::arg("n_samples"))
        .def_readonly("data", &gpppy::GP_data::data)
        .def_readonly("file_path", &gpppy::GP_data::file_path)
        .def_readonly("n_samples", &gpppy::GP_data::n_samples);

    m.def("print", &gpppy::print,
          py::arg("vec"),
          py::arg("start") = 0,
          py::arg("end") = -1,
          py::arg("separator") = " ",
          "Print elements of a vector with optional start, end, and separator parameters");

    //  m.def("print_slice", &print_slice,
    //       py::arg("vec"), py::arg("slice"), py::arg("separator") = " ",
    //       "Print sliced elements of a vector using Python-style slicing notation vec[start:end] with optional separator parameter");
}

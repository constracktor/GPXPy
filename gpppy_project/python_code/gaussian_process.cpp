#include <pybind11/pybind11.h> // PYBIND11_MODULE

namespace py = pybind11;

void init_gpppy(py::module &);
void init_utils(py::module &);

PYBIND11_MODULE(gaussian_process, m)
{
    m.doc() = "Gaussian Process library";

    // NOTE: the order matters. DON'T CHANGE IT!
    init_gpppy(m);
    init_utils(m);
}
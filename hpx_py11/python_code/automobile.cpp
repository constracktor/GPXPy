#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_gpppy(py::module &);
void init_utils(py::module &);

PYBIND11_MODULE(automobile, m)
{
    // Optional docstring
    m.doc() = "Automobile library";

    init_gpppy(m);
    init_utils(m);
}

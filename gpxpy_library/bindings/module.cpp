#include <pybind11/pybind11.h> // PYBIND11_MODULE

namespace py = pybind11;

void init_gpxpy(py::module &);
void init_utils(py::module &);

PYBIND11_MODULE(gpxpy, m)
{
    m.doc() = "GPXPy library";

    // NOTE: the order matters. DON'T CHANGE IT!
    init_gpxpy(m); // adds classes `GP_data`, `Hyperparameters`, `GP` to Python
    init_utils(m); // TODO: comment
}

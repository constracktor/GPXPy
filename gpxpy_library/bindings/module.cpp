#include <pybind11/pybind11.h> // PYBIND11_MODULE

namespace py = pybind11;

void init_gpxpy(py::module&);
void init_utils(py::module&);

/**
 * @brief Python module definition
 *
 * @param m handle for Python module
 */
PYBIND11_MODULE(gpxpy, m)
{
    m.doc() = "GPXPy library";

    // NOTE: order of operations matters

    init_gpxpy(m); // adds classes `GP_data`, `Hyperparameters`, `GP` to Python

    init_utils(m); // adds utility functions `compute_train_tiles`,
                   // `compute_train_tile_size`, `compute_test_tiles`, `print`,
                   // `start_hpx`, `resume_hpx`, `suspend_hpx`, `stop_hpx` to
                   // the module
}

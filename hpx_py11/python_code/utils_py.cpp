#include "../cpp_code/include/automobile_bits/utils_c.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_utils(py::module &m)
{
    m.def("compute_train_tiles", &utils::compute_train_tiles,
          py::arg("n_samples"),
          py::arg("n_tile_size"),
          "Compute number of tiles for training data.");

    m.def("compute_test_tiles", &utils::compute_test_tiles,
          py::arg("m_samples"),
          py::arg("n_tiles"),
          py::arg("n_tile_size"),
          "Compute number of tiles for test data and respective size of test tiles.");
}

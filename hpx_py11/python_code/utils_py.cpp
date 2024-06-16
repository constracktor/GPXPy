#include "../cpp_code/include/automobile_bits/utils_c.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int compute_train_tiles_wrap(int n_samples, int n_tile_size)
{
    if (n_tile_size > 0)
    {
        return utils::compute_train_tiles(n_samples, n_tile_size);
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tile_size.\n");
    }
}

void start_hpx_wrapper(std::vector<std::string> args, int n_cores)
{
    // Add the --hpx:threads argument to the args vector
    args.push_back("--hpx:threads=" + std::to_string(n_cores));

    // Convert std::vector<std::string> to char* array
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(&arg[0]);
    argv.push_back(nullptr);
    int argc = args.size();
    utils::start_hpx_runtime(argc, argv.data());
}

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

void init_utils(py::module &m)
{
    m.def("compute_train_tiles", &compute_train_tiles_wrap,
          py::arg("n_samples"),
          py::arg("n_tile_size"),
          R"pbdoc(
          Compute the number of tiles for training data.

          Parameters:
              n_samples (int): The number of samples.
              n_tile_size (int): The size of each tile.

          Returns:
              tuple: A tuple containing the number of tiles and the adjusted tile size.
          )pbdoc");

    m.def("compute_test_tiles", &utils::compute_test_tiles,
          py::arg("m_samples"),
          py::arg("n_tiles"),
          py::arg("n_tile_size"),
          R"pbdoc(
          Compute the number of tiles for test data and the respective size of test tiles.

          Parameters:
              n_test (int): The number of test samples.
              n_tiles (int): The number of tiles.
              n_tile_size (int): The size of each tile.

          Returns:
              tuple: A tuple containing the number of test tiles and the adjusted tile size.
          )pbdoc");

    m.def("print", &utils::print,
          py::arg("vec"),
          py::arg("start") = 0,
          py::arg("end") = -1,
          py::arg("separator") = " ",
          "Print elements of a vector with optional start, end, and separator parameters");

    m.def("start_hpx", &start_hpx_wrapper, py::arg("args"), py::arg("n_cores")); // Using the wrapper function
    m.def("stop_hpx", &utils::stop_hpx_runtime);

    //  m.def("print_slice", &print_slice,
    //       py::arg("vec"), py::arg("slice"), py::arg("separator") = " ",
    //       "Print sliced elements of a vector using Python-style slicing notation vec[start:end] with optional separator parameter")u
}

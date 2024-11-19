# [GPXPy: Leveraging HPX for Gaussian Processes in Python]()

This repository contains the source code for the GPXPy library, as well as two
reference implementations based on TensorFlow
([GPflow](https://github.com/GPflow/GPflow)) and PyTorch
([GPyTorch](https://github.com/cornellius-gp/gpytorch)).

## Dependencies

GPXPy utilizes three external libraries:
- [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for highly
  parallelized computation
- [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) for CPU-only computation
- [CUDA]() for GPU support (optional)

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack).
A script to install a respective Spack environment `gpxpy` is provided in
[`spack_env`](spack_env).

- Go to the `spack_env` directory and run `setup_spack.sh` in the `GPXPy`
  directory. This script will install Spack in `$HOME`, create a spack
  environment `gpxpy` and install all dependencies.

- For installation on the simcl1 computers of the IPVS institute at the
  University of Stuttgart, use go to the directory
  [`spack_env_simcl1`](spack_env_simcl1) instead.

- If you encounter any issues, we recommend having a look at the
  `setup_spack.sh` and `spack.yaml` files in the respective directories and
  editing them to your needs. The other files in the folder fix minor issues.
  Spack also offers using locally installed packages. The generic install
  script does not look for externally installed programs, as this may result in
  some conflicts. We generally recommend using a new compiler (e.g. GCC for
  CPU-only computation). For GPU support, we recommend using the Clang-Wrapper
  for CUDA instead of NVCC.

## How To Run

To install and run the library on the simcl1 computers of the IPVS institute at
the University of Stuttgart, we recommend using the scripts provided with the
suffix `_simcl1` (if available).

### To run the GPXPy C++ code

- Go to [`gpxpy_library`](gpxpy_library/)
- Run `./compile_cpp.sh` to build the C++ library (CPU-only)
  - Use `./compile_cpp.sh -DGPXPY_WITH_CUBLAS=ON` instead to compile with GPU
    support using cuBLAS
- Set parameters in [`test_cpp/src/execute.cpp`](gpxpy_library/test_cpp/src/execute.cpp)
- Run `./run_cpp.sh` to build and run example

### To run GPXPy with Python

- Go to [`gpxpy_library`](gpxpy_library/)
- Run `./compile_gpxpy.sh` to build the bound Python library
- Set parameters in [`test_gpxpy/config.json`](gpxpy_library/test_gpxpy/config.json)
- Run `./run_gpxpy.sh` to run example

### To run GPflow reference

- Go to [`gpflow_reference`](gpflow_reference/)
- Run `./run_gpflow.sh` to run example

### To run GPflow reference

- Go to [`gpytorch_reference`](gpytorch_reference/)
- Run `./run_gpytorch.sh` to run example

## Project Structure

- [gpxpy_library](./gpxpy_library/): Source code of this library, also compile-
  and test-scripts.
  - [core](./gpxpy_library/core): Source Code for C++ Library
  - [bindings](./gpxpy_library/bindings/): Source Code for Python bindings to
    C++ library.

- [gpflow_reference](./gpflow_reference): Reference implementation with GPflow.
- [gpytorch_reference](./gpytorch_reference): Reference implementation with GPyTorch.

- [data](./data): Files with data for training and testing.

- [spack_env](./spack_env/): Script & Files for Installing Dependencies using
  Spack.
- [spack_env_simcl1](./spack_env_simcl1): Script & Files for Installing
  Dependencies using Spack on simcl1 computers.

- [README.md](./README.md): This file.
- [LICENSE](./LICENSE): MIT License.

## The Team

The GPXPy library is developed by the
[Scientific Computing](https://www.ipvs.uni-stuttgart.de/departments/sc/)
department at IPVS at the University of Stuttgart.
The project is a joined effort of multiple undergraduate, graduate, and PhD
students under the supervision of
[Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/).
We specifically thank the follow contributors:

- [Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/):
  Maintainer and [initial framework](https://doi.org/10.1007/978-3-031-32316-4_5).

- [Maksim Helmann](https://de.linkedin.com/in/maksim-helmann-60b8701b1):
  [Optimization, Python bindings and reference implementations](tbd.).

- [Henrik Möllmann](https://www.linkedin.com/in/moellh/):
  [Accelerator Support for GPXPy: A Task-based Gaussian Process Library in Python]().

## How To Cite

TBD.

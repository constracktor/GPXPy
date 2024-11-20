# [GPXPy: Leveraging HPX for Gaussian Processes in Python]()

This repository contains the source code for the GPXPy library, as well as two
reference implementations based on TensorFlow
([GPflow](https://github.com/GPflow/GPflow)) and PyTorch
([GPyTorch](https://github.com/cornellius-gp/gpytorch)).

## Dependencies

GPXPy utilizes two external libraries:
- [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for asynchronous task-based parallelization
- [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) for CPU-only BLAS computations

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack).
A script to install and setup spack for `GPXPy` is provided in [`spack-repo`](spack-repo).
Spack environment configurations and setup scripts for CPU and GPU use are provided in [`spack-repo/environments`](spack-repo/environments).

## How To Run

### To run the GPXPy C++ code

- Go to [`gpxpy_library`](gpxpy_library/)
- Run `./compile_cpp.sh` to build the C++ library
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

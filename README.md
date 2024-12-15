# [GPXPy: Leveraging HPX for Gaussian Processes in Python]()

This repository contains the source code for the GPXPy library.

## Dependencies

GPXPy utilizes two external libraries:

- [HPX](https://hpx-docs.stellar-group.org/latest/html/index.html) for asynchronous task-based parallelization
- [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) for CPU-only BLAS computations

### Install dependencies

All dependencies can be installed using [Spack](https://github.com/spack/spack).
A script to install and setup spack for `GPXPy` is provided in [`spack-repo`](spack-repo).
Spack environment configurations and setup scripts for CPU and GPU use are provided in [`spack-repo/environments`](spack-repo/environments).

## How To Compile

GPXPy makes use of [CMake presets][1] to simplify the process of configuring
the project.

For example, building and testing this project on a Linux machine is as easy as running the following commands:

```sh
cmake --preset=dev-linux
cmake --build --preset=dev-linux
ctest --preset=dev-linux
```

As a developer, you may create a `CMakeUserPresets.json` file at the root of the project that contains additional
presets local to your machine.

GPXPy can be build with or without Python bindings.
The following options can be set to include / exclude parts of the project:

| Option name                 | Description                                    | Default value   |
|-----------------------------|------------------------------------------------|-----------------|
| GPXPY_BUILD_CORE            | Enable/Disable building of the core library    | ON              |
| GPXPY_BUILD_BINDINGS        | Enable/Disable building of the Python bindings | ON              |
| GPXPY_ENABLE_FORMAT_TARGETS | Enable/disable code formatting helper targets  | ON if top-level |

Respective scripts can be found in this directory.

## How To Run

GPXPy contains several examples. One to run the C++ code, one to run the Python code as well as two
reference implementations based on TensorFlow
([GPflow](https://github.com/GPflow/GPflow)) and PyTorch
([GPyTorch](https://github.com/cornellius-gp/gpytorch)).

### To run the GPXPy C++ code

- Go to [`examples/gpxpy_cpp`](examples/gpxpy_cpp/)
- Set parameters in [`execute.cpp`](examples/gpxpy_cpp/src/execute.cpp)
- Run `./run_gpxpy_cpp.sh` to build and run example

### To run GPXPy with Python

- Go to [`examples/gpxpy_python`](examples/gpxpy_python/)
- Set parameters in [`config.json`](examples/gpxpy_python/config.json)
- Run `./run_gpxpy_python.sh` to run example

### To run GPflow reference

- Go to [`examples/gpflow_reference`](examples/gpflow_reference/)
- Set parameters in [`config.json`](examples/gpflow_reference/config.json)
- Run `./run_gpflow.sh` to run example

### To run GPflow reference

- Go to [`examples/gpytorch_reference`](examples/gpytorch_reference/)
- Set parameters in [`config.json`](examples/gpytorch_reference/config.json)
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
  [Accelerator Support for GPXPy: A Task-based Gaussian Process Library in Python](tbd.).

## How To Cite

TBD.

[1]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html

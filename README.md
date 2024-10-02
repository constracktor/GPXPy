# [GPXPy: Leveraging HPX for Gaussian Processes in Python]()

This repository contains the source code for the GPXPy library, as well as two reference implementations based on Tensorflow ([GPflow](https://github.com/GPflow/GPflow)) and PyTorch ([GPyTorch](https://github.com/cornellius-gp/gpytorch)).

## Dependencies

GPXPy utilizes two external libraries:
- HPX: https://hpx-docs.stellar-group.org/latest/html/index.html
- MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

Both libraries can be installed using [Spack](https://github.com/spack/spack). A script to install a respective Spack environment is provided in [`spack_env`](spack_env).

## How To Run

Note that currently all scripts assume a `gcc/13.2.0` compiler and recent Cmake version to be present. However, the Spack environment and scripts can be modifified to match own compiler availability.
We highly recommend to use the same compiler for HPX and GPXPy.

### Install dependencies

- Run: 'spack_env/setup_spack.sh' in the `GPXPy` directory. This script will install Spack, create an environment for GPXPy and install all dependencies.

### To run the GPXPy C++ code

- Go to [`gpxpy_library`](gpxpy_library/)
- Run `./compile_cpp.sh` to build the C++ library
- Set parameters in [`test_cpp/src/execute.cpp`](gpxpy_library/test_cpp/src/execute.cpp)
- Run `./run_cpp.sh` to build and run example

### To run GPXPy with Python

- Go to [`gpxpy_library`](gpxpy_library/)
- Run `./compile_gpxpy.sh` to build the binded Python library
- Set parameters in [`test_gpxpy/config.json`](gpxpy_library/test_gpxpy/config.json)
- Run `./run_gpxpy.sh` to run example

### To run GPflow reference

- Go to [`gpflow_reference`](gpflow_reference/)
- Run `./run_gpflow.sh` to run example

### To run GPflow reference

- Go to [`gpytorch_reference`](gpytorch_reference/)
- Run `./run_gpytorch.sh` to run example

## The Team

The GPXPy library is developed by the [Scientific Computing](https://www.ipvs.uni-stuttgart.de/departments/sc/) department at IPVS at the University of Stuttgart. The project is a joined effort of multiple undergraduate, graduate, and PhD students under the supervision of [Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/).
We specifically thank the follow contributors:

- [Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/): Maintainer and [intial framework](https://doi.org/10.1007/978-3-031-32316-4_5).

- [Maksim Helmann](https://de.linkedin.com/in/maksim-helmann-60b8701b1): [Optimization, Python bindings and reference implementations](tbd.).

## How To Cite

tbd.

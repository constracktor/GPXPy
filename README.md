
# [GPPPy: Leveraging HPX and BLAS to accelerate Gaussian Processes in Python]()
This repository contains the source code for the master thesis: GPPPy: Leveraging HPX and BLAS to accelerate Gaussian
Processes in Python

## Abstract
Gaussian Processes, also often referred to as Kriging, are a popular regression technique. 
Ranging from geostatistics, and over control theory to AI, GPs are used in many applications. 
Popular software packages in the respective community, e.g., scikit learn, GPy, or GPflow, typically rely on (almost) pure Python and hope to achieve good performance and portability via numpy or tensorflow.
Although in times of big data, problem size is ever-increasing, development typically focuses on additional features. 
Performance-critical optimizations, such as parallelization and performance portability, are outsourced to the backends. 
However, not integrating recent advances in HPC can lead to sub-optimal usage of resources.
The goal of this work is to implement a novel parallel library, called GPPPy (Gaussian Processes Parallel in Python), 
written in C++ that takes advantage of the asynchronous runtime system HPX while still providing the convenience of a Python API. 
The building blocks of GP regression will feature futurized, task-based parallelization as well as a BLAS backend for efficient numerics. 
As a GP application a basic non-linear system identification problem is chosen.

## Initial Implementations
- HPX: https://hpx-docs.stellar-group.org/latest/html/index.html
- MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

Both libraries can be installed by executing [build-hpx.sh](dependencies/build-hpx.sh) and [build-mkl.sh](dependencies/build-mkl.sh).

## Steps to follow to run the project

Run `./run_benchmarks.sh` 

## The Team
Gaussian Processes Parallel in Python (GPPPy) devised by University of Stuttgart grad student [Maksim Helmann](https://de.linkedin.com/in/maksim-helmann-60b8701b1), under the supervision of [Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/) and [Mr. M.Sc. Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/). Intial framework ([Scalability of Gaussian Processes Using Asynchronous Tasks: A Comparison Between HPX and PETSc](https://zenodo.org/records/7535794)) was developed by University of Stuttgart [Mr. M.Sc. Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/).

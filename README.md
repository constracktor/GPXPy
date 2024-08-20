
# [GPPPy: Leveraging HPX and BLAS to accelerate Gaussian Processes in Python]()
This repository contains the source code for the master thesis: GPPPy: Leveraging HPX and BLAS to accelerate Gaussian
Processes in Python

## Abstract
Gaussian processes, often referred to as Kriging, are a popular regression technique. 
They are a widely used alternative to neural networks in various applications, e.g., non-linear system identification. Popular software packages in this domain, such as GPflow and GPyTorch, are based on Python and rely on NumPy or TensorFlow to achieve good performance and portability. Although the problem size continues to grow in the era of big data, the focus of development is primarily on additional features and not on the improvement of parallelization and performance portability.
In this work, we address the aforementioned issues by developing a novel parallel library, GPPPy (Gaussian Processes Parallel in Python). Written in C++, it leverages the asynchronous many-task runtime system HPX, while providing the convenience of a Python API through pybind11. GPPPy includes hyperparameter optimization and the computation of predictions with uncertainty, offering both the marginal variance vector and, if desired, a full posterior covariance matrix computation, hereby making it comparable to existing software packages. We investigate the scaling performance of GPPPy on a dual-socket EPYC 7742 node and compare it against the pure HPX implementation as well as against high-level reference implementations that utilize GPflow and GPyTorch. Our results demonstrate that GPPPy’s performance is directly influenced by the chosen tile size. In addition, we show that there is no runtime overhead when using HPX with pybind11. Compared to GPflow and GPyTorch, our task-based implementation GPPPy is up to 10.5 times faster in our strong scaling benchmarks for prediction with uncertainty computations. Furthermore, GPPPy shows superior parallel efficiency to GPflow and GPyTorch.
Additionally, GPPPy, which only computes predictions with uncertainties, outperforms GPyTorch used with LOVE by a factor of up to 2.8 when using 16 or more cores, despite the latter using an algorithm with superior asymptotic complexity.

## Initial Implementations
- HPX: https://hpx-docs.stellar-group.org/latest/html/index.html
- MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

Both libraries can be installed by executing [build-hpx.sh](dependencies/build-hpx.sh) and [build-mkl.sh](dependencies/build-mkl.sh).

## Steps to follow to run the project

### To run the GPPPy C++ version

- Go to [gpppy_project/cpp_code](gpppy_project/cpp_code)
- Run `./run_cpp.sh` to build the C++ library
- Go to [gpppy_project/cpp_code/tests/src](gpppy_project/cpp_code/tests/src/)
- Set parameters in [../src/execute.py](gpppy_project/cpp_code/tests/src/execute.cpp)
- Go to [../tests](gpppy_project/cpp_code/tests/) and execute `./run.sh`

### To run GPPPy

- Go to [gpppy_project](gpppy_project/)
- Run `./run.sh` to generate the executable
- Go to [gpppy_project/test/](gpppy_project/test/)
- Set parameters in [test/config.json](gpppy_project/test/config.json)
- Run [../tests/execute.py](gpppy_project/test/execute.py) via `python execute.py`

## The Team
Gaussian Processes Parallel in Python (GPPPy) devised by University of Stuttgart grad student [Maksim Helmann](https://de.linkedin.com/in/maksim-helmann-60b8701b1), under the supervision of [Prof. Dr. Dirk Pflüger](https://www.f05.uni-stuttgart.de/en/faculty/contactpersons/Pflueger-00005/) and [Mr. M.Sc. Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/). Intial framework ([Scalability of Gaussian Processes Using Asynchronous Tasks: A Comparison Between HPX and PETSc](https://zenodo.org/records/7535794)) was developed by University of Stuttgart [Mr. M.Sc. Alexander Strack](https://www.ipvs.uni-stuttgart.de/de/institut/team/Strack-00001/).



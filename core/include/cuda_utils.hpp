#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdexcept>

class not_compiled_with_cuda_exception : public std::runtime_error
{
  public:
    not_compiled_with_cuda_exception() :
        std::runtime_error("CUDA is not available because GPXPY has been "
                           "compiled without CUDA.")
    { }
};

#endif  // end of CUDA_UTILS_H

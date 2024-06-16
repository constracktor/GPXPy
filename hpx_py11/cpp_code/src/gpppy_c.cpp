#include "../include/automobile_bits/gpppy_c.hpp"

#include <stdexcept>
#include <cstdio>

namespace gpppy
{
    // Implementation of the Car constructor
    GP_data::GP_data(const std::string &f_path, int n)
    {
        n_samples = n;
        file_path = f_path;
        data = load_data(f_path, n);
    }

    std::vector<double> load_data(const std::string &file_path, int n_samples)
    {
        std::vector<double> _data;
        _data.resize(n_samples);

        FILE *input_file = fopen(file_path.c_str(), "r");
        if (input_file == NULL)
        {
            throw std::runtime_error("Error: File not found: " + file_path);
        }

        // load data
        std::size_t scanned_elements = 0;
        for (int i = 0; i < n_samples; i++)
        {
            // if (fscanf(input_file, "%lf", &data[i]) != 1)
            // {
            //     fclose(training_input_file);
            //     throw std::runtime_error("Error: Failed to read data element at index " + std::to_string(i));
            // }
            scanned_elements += fscanf(input_file, "%lf", &_data[i]); // scanned_elements++;
        }

        fclose(input_file);

        if (scanned_elements != n_samples)
        {
            throw std::runtime_error("Error: Data not correctly read. Expected " + std::to_string(n_samples) + " elements, but read " + std::to_string(scanned_elements));
        }

        return std::move(_data);
    }

    void print(const std::vector<double> &vec, int start, int end, const std::string &separator)
    {
        // Convert negative indices to positive
        if (start < 0)
            start += vec.size();
        if (end < 0)
            end += vec.size() + 1;

        // Ensure the indices are within bounds
        if (start < 0)
            start = 0;
        if (end > vec.size())
            end = vec.size();

        std::cout << vec.size() << std::endl;
        // Validate the range
        if (start >= vec.size() || start >= end)
        {
            std::cerr << "Invalid range" << std::endl;
            return;
        }

        for (int i = start; i < end; i++)
        {
            std::cout << vec[i];
            if (i < end - 1)
            {
                std::cout << separator;
            }
        }
        std::cout << std::endl;
    }
}

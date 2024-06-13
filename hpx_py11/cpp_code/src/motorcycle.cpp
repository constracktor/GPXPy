#include "/home/maksim/simtech/thesis/GPPPy_hpx/hpx_py11/cpp_code/include/automobile_bits/motorcycle.hpp"

#include <iostream>
#include <hpx/hpx.hpp>
#include <hpx/future.hpp>
#include <hpx/iostream.hpp>

#include <hpx/hpx_start.hpp>
#include <hpx/include/post.hpp>

namespace vehicles
{

    Motorcycle::Motorcycle(std::string name)
    {
        _name = name;
    }

    std::string Motorcycle::get_name() const
    {
        return _name;
    }

    void Motorcycle::ride(std::string road) const
    {
        std::cout << "Zoom Zoom on road: " << road << std::endl;
    }
}

namespace university
{

    Student::Student(std::string stud_id)
    {
        _stud_id = stud_id;
    }

    std::string Student::get_stud_id() const
    {
        return _stud_id;
    }

    void Student::do_fut() const
    {
        auto f = hpx::async([]()
                            { return 5; });
        std::cout << "f=" << f.get() << std::endl;
    }

    int Student::add(int i, int j)
    {
        hpx::future<int> fut = hpx::async([i, j]() { return i + j; });

        int result;
        hpx::threads::run_as_hpx_thread([&result, &fut]()
                                        {
                                            result = fut.get(); // Wait for and get the result from the future
                                        });
        return result;
    }

    void Student::start_hpx(int argc, char **argv)
    {
        hpx::start(nullptr, argc, argv);
    }

    void Student::stop_hpx()
    {
        hpx::post([]()
                  { hpx::finalize(); });
        hpx::stop();
    }
}

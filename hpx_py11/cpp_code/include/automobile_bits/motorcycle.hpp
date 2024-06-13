#include <string>

// #include <hpx/hpx_init.hpp>
// #include <hpx/future.hpp>

#ifndef MOTORCYCLE_H
#define MOTORCYCLE_H

namespace vehicles
{

    class Motorcycle
    {

    private:
        /// Name
        std::string _name;

    public:
        /// Constructor
        Motorcycle(std::string name);

        /// Get name
        /// @return Name
        std::string get_name() const;

        /// Ride the bike
        /// @param road Name of the road
        void ride(std::string road) const;
    };

}

namespace university
{

    class Student
    {

    private:
        /// Name
        std::string _stud_id;

    public:
        /// Constructor
        Student(std::string stud_id);

        /// Get name
        /// @return Name
        std::string get_stud_id() const;

        void do_fut() const;

        int add(int i, int j);

        void start_hpx(int argc, char** argv);

        void stop_hpx();
    };

}

#endif

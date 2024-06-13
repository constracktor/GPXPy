#include <automobile>

#include <hpx/include/run_as.hpp>
// #include <hpx/hpx_start.hpp>
#include <hpx/future.hpp>
// #include <hpx/include/post.hpp>
#include <iostream>

void do_futu()
{
    auto f = hpx::async([]()
                        { return 10; });
    std::cout << "f=" << f.get() << std::endl;
}

int main(int argc, char *argv[])
{

    vehicles::Motorcycle m("Yamaha");
    university::Student s("M123");

    std::cout << "Made a car called: " << m.get_name() << std::endl;
    std::cout << "Created student M123: " << s.get_stud_id() << std::endl;

    m.ride("Start Mullholland");

    // Initialize HPX, don't run hpx_main
    // hpx::start(nullptr, argc, argv);
    s.start_hpx(argc, argv);

    // Schedule some functions on the HPX runtime
    // NOTE: run_as_hpx_thread blocks until completion.
    // hpx::threads::run_as_hpx_thread(&m.do_fut);
    hpx::threads::run_as_hpx_thread([&s]()
                                    { s.do_fut(); });

    m.ride("Mitte 1 Mullholland");

    int add_res;
    add_res = s.add(1, 5);
    
    m.ride("Mitte 2 Mullholland");
    
    std::cout << add_res << std::endl;
    // hpx::finalize has to be called from the HPX runtime before hpx::stop
    // hpx::post([]()
    //           { hpx::finalize(); });
    // hpx::stop();
    s.stop_hpx();

    m.ride("Ende Mullholland");

    // hpx::run_as_hpx_thread(&my_other_function, ...);

    return 0;
}

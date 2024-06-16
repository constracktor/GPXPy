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

    int n_train = 10;
    int n_test = 10;
    const int N_CORES = 1; // Set this to the number of threads
    const int tile_size = 10;
    const int n_reg = 100;
    
    int n_tiles = utils::compute_train_tiles(n_train, tile_size);
    std::cout << "n_tiles: " << n_tiles << std::endl;
    auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
    std::cout << "m_tiles: " << result.first << ", m_tile_size: " << result.second << std::endl;
    
    // Create new argc and argv to include the --hpx:threads argument
    std::vector<std::string> args(argv, argv + argc);
    args.push_back("--hpx:threads=" + std::to_string(N_CORES));

    // Convert the arguments to char* array
    std::vector<char*> cstr_args;
    for (auto& arg : args) {
        cstr_args.push_back(const_cast<char*>(arg.c_str()));
    }

    int new_argc = static_cast<int>(cstr_args.size());
    char **new_argv = cstr_args.data();

    gpppy::Kernel_Params kpar;
    std::cout << "lengthscale: " << kpar.lengthscale << std::endl;
    std::cout << kpar.repr() << std::endl;

    gpppy::Hyperparameters hpar;
    std::cout << "lr: " << hpar.learning_rate << std::endl;
    std::cout << hpar.repr() << std::endl;

    vehicles::Motorcycle m("Yamaha");
    university::Student s("M123");


    std::string file_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt";
    gpppy::GP_data training_input(file_path, n_train);
    utils::print(training_input.data, 0, 2, ", ");

    std::string out_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt";
    gpppy::GP_data training_output(out_path, n_train);
    utils::print(training_output.data, 0, 2, ", ");

    std::cout << "Made a car called: " << m.get_name() << std::endl;
    std::cout << "Created student M123: " << s.get_stud_id() << std::endl;

    m.ride("Start Mullholland");


    gpppy::GP gp(training_input.data, training_output.data);
    std::vector<double> training_data = gp.get_training_output();
    utils::print(training_data, 0, 2, ", ");

    // Initialize HPX with the new arguments
    utils::start_hpx_runtime(new_argc, new_argv);

    // vehicles::Motorcycle m("Yamaha");
    // university::Student s("M123");

    // std::cout << "Made a car called: " << m.get_name() << std::endl;
    // std::cout << "Created student M123: " << s.get_stud_id() << std::endl;

    // m.ride("Start Mullholland");

    // // Initialize HPX, don't run hpx_main
    // // hpx::start(nullptr, argc, argv);
    // s.start_hpx(argc, argv);

    // Schedule some functions on the HPX runtime
    // NOTE: run_as_hpx_thread blocks until completion.
    // hpx::threads::run_as_hpx_thread(&m.do_fut);
    hpx::threads::run_as_hpx_thread([&s]()
                                    { s.do_fut(); });

    m.ride("Mitte 1 Mullholland");

    int add_res;
    add_res = s.add(1, 1000);
    
    m.ride("Mitte 2 Mullholland");
    
    std::cout << add_res << std::endl;
    // hpx::finalize has to be called from the HPX runtime before hpx::stop
    // hpx::post([]()
    //           { hpx::finalize(); });

    // int add_res123;
    // add_res123 = s.add(1, 5000);

    // std::cout << add_res123 << std::endl;

    // hpx::stop();
    utils::stop_hpx_runtime();

    m.ride("Ende Mullholland");

    // hpx::run_as_hpx_thread(&my_other_function, ...);

    return 0;
}

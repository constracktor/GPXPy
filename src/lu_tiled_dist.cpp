#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/util/unused.hpp>

#include <boost/shared_array.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
inline std::size_t idx(std::size_t i, std::size_t j, std::size_t s)
{
    return i * s + j;
}

inline std::size_t locidx(
    std::size_t i, std::size_t j, std::size_t T, std::size_t nl)
{
    std::size_t tnl = T / nl;
    if (T % nl > 0)
        ++tnl;
    return i / tnl;
}

///////////////////////////////////////////////////////////////////////////////
struct partition_data
{
private:
    typedef hpx::serialization::serialize_buffer<double> buffer_type;

public:
    partition_data()
      : n_(0)
      , t_(0)
      , i_(0)
      , j_(0)
    {
    }

    // Create a new (uninitialized) partition of the given size.
    partition_data(std::size_t n)
      : data_(
            std::allocator<double>().allocate(n * n), n * n, buffer_type::take)
      , n_(n)
      , t_(0)
      , i_(0)
      , j_(0)
    {
    }

    // Create a new (initialized) partition of the given size.
    partition_data(std::size_t n, std::size_t t, std::size_t i, std::size_t j)
      : data_(
            std::allocator<double>().allocate(n * n), n * n, buffer_type::take)
      , n_(n)
      , t_(t)
      , i_(i)
      , j_(j)
    {
        srand(idx(i, j, n));
        for (std::size_t k = 0; k != n * n; ++k)
            data_[k] = (double) ((rand() % 2000) - 1000) / 100;
    }

    partition_data(partition_data const& base)
      : data_(base.data_.data(), base.size(), buffer_type::copy)
      , n_(base.n_)
      , t_(base.t_)
      , i_(base.i_)
      , j_(base.j_)
    {
    }

    double& operator[](std::size_t idx)
    {
        return data_[idx];
    }
    double operator[](std::size_t idx) const
    {
        return data_[idx];
    }

    std::size_t size() const
    {
        return n_ * n_;
    }

    std::size_t dim() const
    {
        return n_;
    }

    std::size_t nt() const
    {
        return t_;
    }

    std::size_t pos_i() const
    {
        return i_;
    }

    std::size_t pos_j() const
    {
        return j_;
    }

private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& data_& n_& t_& i_& j_;
    }

private:
    buffer_type data_;
    std::size_t n_;
    std::size_t t_;
    std::size_t i_;
    std::size_t j_;
};

std::ostream& operator<<(std::ostream& os, partition_data const& c)
{
    os << "== block i(" << c.pos_i() << ") j(" << c.pos_j() << ")" <<  std::endl;
    for (std::size_t i = 0; i != c.dim(); ++i)
    {
        for (std::size_t j = 0; j != c.dim(); ++j)
        {
            os << c.pos_i() * c.dim() + i << " " << c.pos_j() * c.dim() + j << " " << c[i * c.dim() + j]
               << std::endl;
        }
    }
    return os;
}

///////////////////////////////////////////////////////////////////////////////
// This is the server side representation of the data. We expose this as a HPX
// component which allows for it to be created and accessed remotely through
// a global address (hpx::id_type).
struct partition_server : hpx::components::component_base<partition_server>
{
    // construct new instances
    partition_server() {}

    partition_server(partition_data const& data)
      : data_(data)
    {
    }

    partition_server(std::size_t n, std::size_t t, std::size_t i, std::size_t j)
      : data_(n, t, i, j)
    {
    }

    // Access data.
    partition_data get_data() const
    {
        return data_;
    }

    // Every member function which has to be invoked remotely needs to be
    // wrapped into a component action. The macro below defines a new type
    // 'get_data_action' which represents the (possibly remote) member function
    // partition::get_data().
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(
        partition_server, get_data, get_data_action);

private:
    partition_data data_;
};

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<partition_server> partition_server_type;
HPX_REGISTER_COMPONENT(partition_server_type, partition_server);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef partition_server::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action);

///////////////////////////////////////////////////////////////////////////////
// This is a client side helper class allowing to hide some of the tedious
// boilerplate.
struct partition : hpx::components::client_base<partition, partition_server>
{
    typedef hpx::components::client_base<partition, partition_server> base_type;

    partition() {}

    // Create new component on locality 'where' and initialize the held data
    partition(hpx::id_type where, std::size_t n, std::size_t t, std::size_t i, std::size_t j)
      : base_type(hpx::new_<partition_server>(where, n, t, i, j))
    {
    }

    // Create a new component on the locality co-located to the id 'where'. The
    // new instance will be initialized from the given partition_data.
    partition(hpx::id_type where, partition_data const& data)
      : base_type(hpx::new_<partition_server>(hpx::colocated(where), data))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    partition(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    partition(hpx::future<partition>&& c)
      : base_type(std::move(c))
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<partition_data> get_data() const
    {
        partition_server::get_data_action act;
        return hpx::async(act, get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
struct stepper
{
    // Our data for one time step
    typedef std::vector<partition> space;

    static partition_data inv_core(partition_data const& A_)
    {
        double tmp;
        partition_data A(A_);
        partition_data v(A_.dim());

        for (int i = 0; i < A.dim(); ++i)
        {
            for (int j = 0; j < A.dim(); ++j)
            {
                v[i * A.dim() + j] = 0;
            }
            v[i * A.dim() + i] = 1;
        }
        for (int k = 0; k < A.dim(); ++k)
        {
            tmp = A[k * A.dim() + k];
            for (int j = 0; j < A.dim(); ++j)
            {
                v[k * A.dim() + j] /= tmp;
                A[k * A.dim() + j] /= tmp;
            }

            // can be parallalized
            for (int i = 0; i < k; ++i)
            {
                tmp = A[i * A.dim() + k];
                for (int j = 0; j < A.dim(); ++j)
                {
                    v[i * A.dim() + j] -= tmp * v[k * A.dim() + j];
                    A[i * A.dim() + j] -= tmp * A[k * A.dim() + j];
                }
            }
            for (int i = k + 1; i < A.dim(); ++i)
            {
                tmp = A[i * A.dim() + k];
                for (int j = 0; j < A.dim(); ++j)
                {
                    v[i * A.dim() + j] -= tmp * v[k * A.dim() + j];
                    A[i * A.dim() + j] -= tmp * A[k * A.dim() + j];
                }
            }
        }
        return v;
    }

    //A = A * B
    static partition_data pmm_core(partition_data const& A, partition_data const& B)
    {
        partition_data r(A);

        // i and j can be parallalized
        for (int i = 0; i < A.dim(); ++i)
        {
            for (int j = 0; j < A.dim(); ++j)
            {
                r[i * A.dim() + j] = 0;
                for (int k = 0; k < A.dim(); ++k)
                {
                    r[i * A.dim() + j] += A[i * A.dim() + k] * B[k * A.dim() + j];
                }
            }
        }
        return r;
    }

    //C = C - A * B
    static partition_data pmm_d_core(
        partition_data const& A, partition_data const& B, partition_data const& C)
    {
        partition_data r(C);

        // i and j can be parallalized
        for (int i = 0; i < A.dim(); ++i)
        {
            for (int j = 0; j < A.dim(); ++j)
            {
                for (int k = 0; k < A.dim(); ++k)
                {
                    r[i * A.dim() + j] -= A[i * A.dim() + k] * B[k * A.dim() + j];
                }
            }
        }
        return r;
    }

    static partition inv_part(partition const& A_p, partition const& inv_p)
    {
        using hpx::dataflow;
        using hpx::util::unwrapping;

        hpx::shared_future<partition_data> A_data = A_p.get_data();
        return dataflow(hpx::launch::async,
            unwrapping([inv_p](partition_data const& A) -> partition {
                partition_data m_inv = stepper::inv_core(A);
                return partition(inv_p.get_id(), m_inv);
            }),
            A_data);
    }

    static partition pmm_part(partition const& A_p, partition const& B_p)
    {
        using hpx::dataflow;
        using hpx::util::unwrapping;

        hpx::shared_future<partition_data> A_data = A_p.get_data();
        hpx::shared_future<partition_data> B_data = B_p.get_data();
        return dataflow(hpx::launch::async,
            unwrapping([A_p](partition_data const& A,
                           partition_data const& B) -> partition {
                partition_data r = stepper::pmm_core(A, B);
                return partition(A_p.get_id(), r);
            }),
            A_data, B_data);
    }

    static partition pmm_d_part(
        partition const& A_p, partition const& B_p, partition const& C_p)
    {
        using hpx::dataflow;
        using hpx::util::unwrapping;

        hpx::shared_future<partition_data> A_data = A_p.get_data();
        hpx::shared_future<partition_data> B_data = B_p.get_data();
        hpx::shared_future<partition_data> C_data = C_p.get_data();
        return dataflow(hpx::launch::async,
            unwrapping([C_p](partition_data const& A, partition_data const& B,
                           partition_data const& C) -> partition {
                partition_data r = stepper::pmm_d_core(A, B, C);
                return partition(C_p.get_id(), r);
            }),
            A_data, B_data, C_data);
    }

    // do all the work on 'np' partitions, 'nx' data points each, for 'nt'
    // time steps
    space do_lu(std::size_t T, std::size_t N, bool print_matrices);
};

HPX_PLAIN_ACTION(stepper::inv_part, inv_part_action);
HPX_PLAIN_ACTION(stepper::pmm_part, pmm_part_action);
HPX_PLAIN_ACTION(stepper::pmm_d_part, pmm_d_part_action);

///////////////////////////////////////////////////////////////////////////////
// do all the work on 'np' partitions, 'nx' data points each, for 'nt'
// time steps
stepper::space stepper::do_lu(std::size_t T, std::size_t N, bool print_matrices)
{
    using hpx::dataflow;

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();    // Number of localities

    space tiles;
    tiles.resize(T * T);

    space invs;
    invs.resize(T);

    for (std::size_t i = 0; i != T; ++i)
        for (std::size_t j = 0; j != T; ++j)
            tiles[idx(i, j, T)] =
                partition(localities[locidx(i, j, T, nl)], N, T, i, j);

    if (print_matrices) {
        std::cout << "== a == " << std::endl;
        for (std::size_t i = 0; i != T * T; ++i)
        {
            tiles[i].get_data().wait();
            std::cout << tiles[i].get_data().get();
        }
        std::cout << "== a == " << std::endl;
    }

    for (std::size_t i = 0; i != T; ++i)
        invs[i] = partition(localities[locidx(i, 0, T, nl)], N, T, i, 0);

    inv_part_action act_inv;
    pmm_part_action act_pmm;
    pmm_d_part_action act_pmm_d;

    for (std::size_t k = 0; k < T - 1; ++k)
    {
        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;
        auto Op =
            hpx::util::bind(act_inv, localities[locidx(k, 0, T, nl)], _1, _2);
        invs[k] =
            dataflow(hpx::launch::async, Op, tiles[idx(k, k, T)], invs[k]);
        for (std::size_t i = k + 1; i < T; ++i)
        {
            using hpx::util::placeholders::_1;
            using hpx::util::placeholders::_2;
            auto Op = hpx::util::bind(
                act_pmm, localities[locidx(k, 0, T, nl)], _1, _2);
            tiles[idx(i, k, T)] =
                dataflow(hpx::launch::async, Op, tiles[idx(i, k, T)], invs[k]);
            for (std::size_t j = k + 1; j < T; ++j)
            {
                using hpx::util::placeholders::_1;
                using hpx::util::placeholders::_2;
                using hpx::util::placeholders::_3;
                auto Op = hpx::util::bind(
                    act_pmm_d, localities[locidx(i, j, T, nl)], _1, _2, _3);
                tiles[idx(i, j, T)] =
                    dataflow(hpx::launch::async, Op, tiles[idx(i, k, T)],
                        tiles[idx(k, j, T)], tiles[idx(i, j, T)]);
            }
        }
    }

    // Return the LU factorization
    return tiles;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::uint64_t N = vm["N"].as<std::uint64_t>();
    std::uint64_t T = vm["T"].as<std::uint64_t>();

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();    // Number of localities

    if (N * N < nl)
    {
        std::cout << "The number of tiles should not be smaller than "
                     "the number of localities"
                  << std::endl;
        return hpx::finalize();
    }

    // Create the stepper object
    stepper step;

    // Measure execution time.
    std::uint64_t t = hpx::util::high_resolution_clock::now();

    // Execute nt time steps on nx grid points and print the final solution.
    stepper::space solution = step.do_lu(T, N, vm.count("print-matrices"));
    for (std::size_t i = 0; i != T * T; ++i)
        solution[i].get_data().wait();

    std::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;
    std::cout << elapsed / 1e9 << std::endl;

    // Print the final solution
    if (vm.count("print-matrices"))
    {
        std::cout << "== lu == " << std::endl;
        for (std::size_t i = 0; i != T * T; ++i)
        {
            std::cout << solution[i].get_data().get();
        }
        std::cout << "== lu == " << std::endl;
    }
    std::uint64_t const num_worker_threads = hpx::get_num_worker_threads();
    hpx::future<std::uint32_t> locs = hpx::get_num_localities();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()(
        "print-matrices", "print generated A and computed LU (default: false)")("N",
        value<std::uint64_t>()->default_value(10),
        "Dimension of the submatrices")("T",
        value<std::uint64_t>()->default_value(10),
        "Number of subblocks in each dimension");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

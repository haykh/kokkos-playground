#include <Kokkos_Core.hpp>

#include <chrono>
#include <iostream>

namespace math = Kokkos;

struct Particles {
  const std::size_t     npart;
  Kokkos::View<double*> x;

  Particles(const std::size_t& _npart) : npart { _npart }, x { "x", _npart } {}

  void Init() {
    auto _x = x;
    Kokkos::parallel_for(
      "init", npart, KOKKOS_LAMBDA(const std::size_t p) { _x(p) = (double)p / (double)npart; });
  }
};

// template <typename T>
// class myallocator {
// public:
// using value_type = T;

// T* allocate(std::size_t n) {
// T* ptr;
// Kokkos::malloc(&ptr, n * sizeof(T));
// return ptr;
//}

// void deallocate(T* ptr, std::size_t n) {
// Kokkos::free(ptr);
//}
//};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    int                    nblocks  = 0;
    int                    nthreads = 1024;
    // ExecutionSpace::execution_space::concurrency();
    // int max_threads = std::thread::hardware_concurrency();

    std::vector<Particles> species {
      Particles(100000),  Particles(1000000), Particles(10000),
      Particles(1000000), Particles(100000),  Particles(1000000)
    };

    Kokkos::View<double*> EB("EB", 1000);

    // printf("max_threads = %d\n", max_threads);
    //  define number of threads in a block

    // NTHREADS = 1024;

    // find number end blocks to launch
    // number_of_blocks = for spec in species - ceil(spec.npart/NTHREADS)
    for (auto& spec : species) {
      nblocks += std::ceil(static_cast<float>(spec.npart) / nthreads);
    }

    Kokkos::View<int* [2]> starting_index("starting_index", nblocks);
    auto                   starting_index_host = Kokkos::create_mirror_view(starting_index);

    //  initialize starting_index with (species_index, particle_index)
    int                    block_index         = 0;
    for (auto& spec : species) {
      for (int i = 0; i < spec.npart; i += nthreads) {
        starting_index_host(block_index, 0) = &spec - &species[0];
        starting_index_host(block_index, 1) = i;
        block_index++;
      }
    }
    Kokkos::deep_copy(starting_index, starting_index_host);

    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type member_type;
    Kokkos::TeamPolicy policy(nblocks, nthreads);

    auto               time1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        int member_rank = team_member.league_rank();
        int begin       = member_rank;
        int end         = begin + 1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nthreads), [&](const int& i) {
          int block_index = begin * nthreads + i;
          if (block_index < nblocks) {
            // Get the starting index for this block
            int species_index  = starting_index(block_index, 0);
            int particle_index = starting_index(block_index, 1);

            if (particle_index < species[species_index].npart) {
              const auto x_coord  = species[species_index].x(particle_index);
              const auto eb_ind   = (int)(1000 * x_coord);
              const auto EB_field = EB(eb_ind);
              species[species_index].x(particle_index) += 0.01 * EB_field + math::sin(x_coord);
            }
          }
        });
      });
    Kokkos::fence();
    auto time2 = std::chrono::high_resolution_clock::now();
    printf("time = %f\n", std::chrono::duration<double>(time2 - time1).count());

    auto time3 = std::chrono::high_resolution_clock::now();
    for (auto& spec : species) {
      Kokkos::parallel_for(
        "loop_species", spec.npart, KOKKOS_LAMBDA(const std::size_t p) { spec.x(p) += 0.1; });
    }
    Kokkos::fence();
    auto time4 = std::chrono::high_resolution_clock::now();
    printf("time = %f\n", std::chrono::duration<double>(time4 - time3).count());
  } catch (std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}

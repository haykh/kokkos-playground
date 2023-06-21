#include <Kokkos_Core.hpp>

#include <iostream>

struct Species {
  std::size_t npart;

  Species(const std::size_t& _npart) : npart { _npart } {}
};

struct Particles : public Species {
  Kokkos::View<double*> x;

  Particles(const std::size_t& _npart) : Species { _npart }, x { "x", _npart } {}
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    using Kokkos::TeamPolicy;
    using Kokkos::parallel_for;
    using Kokkos::TeamThreadRange;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    
    Kokkos::Timer timer;

    int nblocks = 0;
    int nthreads = 1024;//ExecutionSpace::execution_space::concurrency();
    // int max_threads = std::thread::hardware_concurrency();

    std::vector<Particles> species { Particles(100000), Particles(1000000), Particles(10000), Particles(1000000), Particles(100000), Particles(1000000) };



    //printf("max_threads = %d\n", max_threads);
    // define number of threads in a block

    //NTHREADS = 1024;

    // find number end blocks to launch
    //number_of_blocks = for spec in species - ceil(spec.npart/NTHREADS)


    for (auto& spec : species) {
      nblocks += std::ceil(static_cast<float>(spec.npart)/nthreads);
    }

    Kokkos::View<int*[2]> starting_index("starting_index", nblocks);
    auto starting_index_host = Kokkos::create_mirror_view(starting_index);

    Kokkos::deep_copy(starting_index_host, starting_index); 
    // initialize starting_index with (species_index, particle_index)
    int block_index = 0;
    for (auto& spec : species) {
      for (int i = 0; i < spec.npart; i += nthreads) {
        starting_index_host(block_index, 0) = &spec - &species[0];
        starting_index_host(block_index, 1) = i;
        block_index++;
      }
    }
    Kokkos::deep_copy(starting_index, starting_index_host);

    typedef TeamPolicy<ExecutionSpace>::member_type member_type;
    Kokkos::TeamPolicy policy(nblocks, nthreads);

    double time1 = timer.seconds();
    Kokkos::parallel_for (policy, KOKKOS_LAMBDA (member_type team_member) {
      int member_rank = team_member.league_rank();
      int begin = member_rank;
      int end = begin + 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nthreads), [=] (const int& i) {
        int block_index = begin * nthreads + i;
        if (block_index < nblocks) {
          // Get the starting index for this block
          int species_index = starting_index(block_index,0);
          int particle_index = starting_index(block_index,1);

          if (particle_index < species[species_index].npart) {
            species[species_index].x(particle_index) += 0.01;
          }
        }
      }
      );
    }
    );
    double time2 = timer.seconds();
    printf("time = %f\n", time2 - time1);
    
    double time3 = timer.seconds();
    for (auto& spec : species) {
      Kokkos::parallel_for(
        "loop_species", spec.npart, KOKKOS_LAMBDA(const std::size_t p) { spec.x(p) += 0.1; });
    }
    double time4 = timer.seconds();
    printf("time = %f\n", time4 - time3);
    // Kokkos::parallel_for(Kokkos::MDRangePolicy<DevExecutionSpace, Kokkos::Rank<2, Iterate::Right>>({0, 0}, {species.size(), }))
  }
  Kokkos::finalize();
  return 0;
}

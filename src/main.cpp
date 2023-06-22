#include <Kokkos_Core.hpp>

#include <chrono>
#include <iostream>

namespace math = Kokkos;

struct Particles {
public:
  std::size_t     npart;
  Kokkos::View<double*> x;

  Particles() = default;

  Particles(const std::size_t& _npart) : npart { _npart }, x { "x", _npart } {Init();}

  // Particles(Particles &p) = default;

  ~Particles() = default;

  // Particles& operator=(const Particles& other) {
  //   if (this != &other) {
  //     std::size_t temp_npart = other.npart;
  //     this->npart = temp_npart;
  //     this->x = other.x;
  //   }
  //   return *this;
  // }

  void Init() {
  // int _npart = this->npart;
  // auto _x = this->x;
  Kokkos::parallel_for(
    "init", npart, KOKKOS_CLASS_LAMBDA(const std::size_t p) { x(p) = (double)p / (double)npart; });
  }
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    int                    nblocks  = 0;
    int                    nthreads = 1024;
    // ExecutionSpace::execution_space::concurrency();
    // int max_threads = std::thread::hardware_concurrency();

    // using species_allocator = Kallocator<Particles>;
    // std::vector<Particles*> species {
    //   Particles(100000),  Particles(1000000), Particles(10000),
    //   Particles(1000000), Particles(100000),  Particles(1000000)
    // };


    Kokkos::View<Particles *> species("h_species",2);

    auto h_species = Kokkos::create_mirror_view(species);

    for (int i = 0; i < 2; i++) {
      h_species(i) = Particles(10000);
    }

    Kokkos::deep_copy(species, h_species);

    Kokkos::View<double*> EB("EB", 1000);

    for (std::size_t i = 0; i < h_species.extent(0); ++i) {
      auto& spec = h_species(i);
      nblocks += std::ceil(static_cast<float>(spec.npart) / nthreads);
    }

    printf("flag1 \n");
    Kokkos::View<int* [2]> starting_index("starting_index", nblocks);
    auto                   starting_index_host = Kokkos::create_mirror_view(starting_index);

    //  initialize starting_index with (species_index, particle_index)
    int block_index = 0;
    for (std::size_t i = 0; i < h_species.extent(0); ++i) {
      auto& spec = h_species(i);
      for (int j = 0; j < spec.npart; j += nthreads) {
        starting_index_host(block_index, 0) = i;
        starting_index_host(block_index, 1) = j;
        block_index++;
      }
    }
    printf("flag2 \n"); 

    Kokkos::deep_copy(starting_index, starting_index_host);

    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type member_type;
    Kokkos::TeamPolicy policy(nblocks, nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        int member_rank = team_member.league_rank();
        int begin       = member_rank;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nthreads), [&](const int& i) {
          int block_index = begin * nthreads + i;
          if (block_index < nblocks) {
            // Get the starting index for this block
            int species_index  = starting_index(block_index, 0);
            int particle_index = starting_index(block_index, 1);

            // if (particle_index < species[species_index].npart) {
            //   const auto x_coord  = species[species_index].x(particle_index);
            //   const auto eb_ind   = (int)(1000 * x_coord);
            //   const auto EB_field = EB(eb_ind);
            //    species[species_index].x(particle_index) += 0.01 * EB_field + math::sin(x_coord);
            // }
          }
        });
      });
    Kokkos::fence();

    printf("flag3 \n"); 

    // add up all element 
    for (std::size_t i = 0; i < h_species.extent(0); ++i) {
      auto& spec = h_species(i);
      double g_sum = 0.0;
      Kokkos::parallel_reduce(
        "loop_species", spec.npart, KOKKOS_LAMBDA(const std::size_t p, double& sum) {
          sum += spec.x(p);
        }, g_sum);
      printf("sum = %f\n", g_sum);
    }

    auto time2 = std::chrono::high_resolution_clock::now();
    printf("time = %f\n", std::chrono::duration<double>(time2 - time1).count());

    auto time3 = std::chrono::high_resolution_clock::now();
    // for (auto& spec : species) {
    for (std::size_t i = 0; i < h_species.extent(0); ++i) {
      auto& spec = h_species(i);
      Kokkos::parallel_for(
        "loop_species", spec.npart, KOKKOS_LAMBDA(const std::size_t p) { spec.x(p) += 0.1; });
    }
    Kokkos::fence();

    for (std::size_t i = 0; i < h_species.extent(0); ++i) {
      auto& spec = h_species(i);
      double g_sum = 0.0;
      Kokkos::parallel_reduce(
        "loop_species", spec.npart, KOKKOS_LAMBDA(const std::size_t p, double& sum) {
          sum += spec.x(p);
        }, g_sum);
      printf("sum = %f\n", g_sum);
    }


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

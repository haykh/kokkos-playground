#include "playground.hpp"

#include <Kokkos_Core.hpp>

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#include <iostream>

auto Playground() -> void {
  constexpr float      x1min_g = 0.0f, x1max_g = 1.0f;
  Kokkos::View<float*> x1("x1", 10000);
  Kokkos::View<float*> ux1("ux1", 10000);
  Kokkos::View<short*> tag("tag", 10000);

  float x1min, x1max;
#ifdef MPI_ENABLED
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  x1min = x1min_g + (x1max_g - x1min_g) * mpi_rank / mpi_size;
  x1max = x1min_g + (x1max_g - x1min_g) * (mpi_rank + 1) / mpi_size;
#else
  x1min = x1min_g;
  x1max = x1max_g;
#endif

  std::cout << "Hello, world!\n";
}
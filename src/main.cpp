#include "playground.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>

#ifdef ADIOS2_ENABLED
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
#endif

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

auto Initialize(int argc, char* argv[]) -> void;
auto Finalize() -> void;

auto main(int argc, char* argv[]) -> int {
  try {
    Initialize(argc, argv);
    Playground();
  } catch (std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    Finalize();
    return 1;
  }
  Finalize();
  return 0;
}

auto Initialize(int argc, char* argv[]) -> void {
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize(argc, argv);
  }
#ifdef MPI_ENABLED
  MPI_Init(&argc, &argv);
#endif
}

auto Finalize() -> void {
#ifdef MPI_ENABLED
  MPI_Finalize();
#endif
  if (Kokkos::is_initialized()) {
    Kokkos::finalize();
  }
}

#include "playground.hpp"

#include <Kokkos_Core.hpp>

#if defined(ADIOS2_ENABLED)
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
#endif

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

auto Initialize(int argc, char* argv[]) -> void;
auto Finalize() -> void;

auto main(int argc, char* argv[]) -> int {
  Initialize(argc, argv);
  try {
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
  Kokkos::initialize(argc, argv);
#if defined(MPI_ENABLED)
  MPI_Init(&argc, &argv);
#endif
}

auto Finalize() -> void {
#if defined(MPI_ENABLED)
  MPI_Finalize();
#endif
  Kokkos::finalize();
}
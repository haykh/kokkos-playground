#include "playground.hpp"

#include <Kokkos_Core.hpp>

#ifdef ADIOS2_ENABLED
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
#endif

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#include <string>

auto Initialize(int argc, char* argv[]) -> void;
auto Finalize() -> void;

void help(const std::string& name) {
  std::cerr << "Usage: " << name << " [read|write]\n";
  throw std::invalid_argument("Invalid arguments");
}

auto main(int argc, char* argv[]) -> int {
  Initialize(argc, argv);
  try {
    if (argc != 2) {
      help(argv[0]);
    }
    const auto action = std::string(argv[1]);
    if (action != "read" && action != "write") {
      help(argv[0]);
    }
    Playground(action);
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
#ifdef MPI_ENABLED
  MPI_Init(&argc, &argv);
#endif
}

auto Finalize() -> void {
#ifdef MPI_ENABLED
  MPI_Finalize();
#endif
  Kokkos::finalize();
}

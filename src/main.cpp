#include <Kokkos_Core.hpp>

#include <chrono>
#include <iostream>

namespace math = Kokkos;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
  } catch (std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}

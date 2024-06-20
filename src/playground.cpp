#include "playground.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>

auto Playground() -> void {
  constexpr int                           N = 100;
  Kokkos::View<double*, Kokkos::HIPSpace> A { "A", N };
  Kokkos::parallel_for(
    Kokkos::RangePolicy<Kokkos::HIP>(0, N),
    KOKKOS_LAMBDA(int i) {
      A(i) = 1.0; // / (static_cast<double>(i + 1) * static_cast<double>(i + 1));
    });
  // double sum = 0.0;
  // Kokkos::parallel_reduce(
  //   N,
  //   KOKKOS_LAMBDA(int i, double& lsum) { lsum += A(i); },
  //   sum);
  // std::cout << "pi = " << Kokkos::sqrt(sum * 6.0) << '\n';
  // std::cout << "Hello, world!\n";
}

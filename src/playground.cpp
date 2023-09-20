#include "playground.hpp"

#include "timer.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <iostream>

auto Playground() -> void {
  auto random_pool = RandomNumberPool_t(RandomSeed);

  constexpr std::size_t ngh      = 2;
  const std::size_t     npart    = (std::size_t)(1e7);
  const std::size_t     maxnpart = (std::size_t)(1e8);
  const std::size_t     nx1 = 5000, nx2 = 5000;

  Kokkos::View<int*>   i1("x1", maxnpart);
  Kokkos::View<int*>   i2("x2", maxnpart);
  Kokkos::View<short*> tag("tag", maxnpart);

  Kokkos::View<float*> dx1("x1", maxnpart);
  Kokkos::View<float*> dx2("x2", maxnpart);

  Kokkos::View<real_t*> ux1("ux1", maxnpart);
  Kokkos::View<real_t*> ux2("ux2", maxnpart);
  Kokkos::View<real_t*> ux3("ux3", maxnpart);

  Kokkos::View<real_t** [6]> fld("fld", nx1 + 2 * ngh, nx2 + 2 * ngh);
  const real_t               coeff { 0.25 };

  timer(
    [&]() {
      Kokkos::parallel_for(
        "init_f",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 },
                                               { nx1 + 2 * ngh, nx2 + 2 * ngh }),
        InitFlds_kernel(fld, random_pool));
    },
    "init_f")();

  timer(
    [&]() {
      Kokkos::parallel_for(
        "init_p",
        npart,
        InitPrtls_kernel(nx1, nx2, i1, i2, dx1, dx2, ux1, ux2, ux3, random_pool));
    },
    "init_p")();

  timer(
    [&]() {
      Kokkos::parallel_for(
        "push_p",
        npart,
        Push_kernel(ngh, coeff, fld, i1, i2, dx1, dx2, ux1, ux2, ux3));
    },
    "push_p",
    10)();

  timer(
    [&]() {
      Kokkos::parallel_for("bc_p", npart, PrtlsBC_kernel(nx1, nx2, i1, i2));
    },
    "bc_p",
    10)();

  timer(
    [&]() {
      auto energy = 0.0f;
      Kokkos::parallel_reduce(
        "energy_p",
        npart,
        KOKKOS_LAMBDA(index_t p, real_t & nrg) {
          nrg += math::sqrt(
            1.0f + ux1(p) * ux1(p) + ux2(p) * ux2(p) + ux3(p) * ux3(p));
        },
        energy);
      std::cout << "energy: " << energy << std::endl;
    },
    "energy_p")();
}
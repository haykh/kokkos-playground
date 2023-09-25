#include "playground.hpp"

#include "timer.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <stdio.h>

#include <iostream>

auto countNdead(Kokkos::View<short*> tag, std::size_t npart, std::size_t maxnpart)
  -> std::pair<std::size_t, std::size_t> {
  Kokkos::View<std::size_t> ndead_1 { "ndead1" };
  Kokkos::View<std::size_t> ndead_2 { "ndead2" };
  Kokkos::parallel_for(
    "count_p",
    maxnpart,
    KOKKOS_LAMBDA(index_t p) {
      if (tag(p) == tag::dead) {
        if (p >= npart) {
          Kokkos::atomic_fetch_add(&ndead_2(), 1);
        } else {
          Kokkos::atomic_fetch_add(&ndead_1(), 1);
        }
      }
    });
  auto ndead_1_h = Kokkos::create_mirror_view(ndead_1);
  auto ndead_2_h = Kokkos::create_mirror_view(ndead_2);
  Kokkos::deep_copy(ndead_1_h, ndead_1);
  Kokkos::deep_copy(ndead_2_h, ndead_2);
  return { ndead_1_h(), ndead_2_h() };
}

auto Playground() -> void {
  auto random_pool = RandomNumberPool_t(RandomSeed);

  constexpr std::size_t ngh      = 2;
  const std::size_t     npart    = (std::size_t)(1e5);
  const std::size_t     maxnpart = (std::size_t)(1e6);
  const std::size_t     nx1 = 500, nx2 = 500;

  Kokkos::View<int*>   i1("i1", maxnpart);
  Kokkos::View<int*>   i2("i2", maxnpart);
  Kokkos::View<float*> dx1("x1", maxnpart);
  Kokkos::View<float*> dx2("x2", maxnpart);

  Kokkos::View<real_t*> ux1("ux1", maxnpart);
  Kokkos::View<real_t*> ux2("ux2", maxnpart);
  Kokkos::View<real_t*> ux3("ux3", maxnpart);

  Kokkos::View<short*> tag("tag", maxnpart);

  Kokkos::View<real_t** [6]> fld("fld", nx1 + 2 * ngh, nx2 + 2 * ngh);
  const real_t               coeff { 0.1 };

  Kokkos::parallel_for(
    "init_f",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 },
                                           { nx1 + 2 * ngh, nx2 + 2 * ngh }),
    InitFlds_kernel(fld, random_pool));

  Kokkos::parallel_for(
    "init_p",
    npart,
    InitPrtls_kernel(i1, i2, dx1, dx2, ux1, ux2, ux3, tag, nx1, nx2, random_pool));

  {
    auto [nd1, nd2] = countNdead(tag, npart, maxnpart);
    printf("Dead < npart : npart=%ld dead=%ld %%=%f",
           npart,
           nd1,
           100.0 * (double)dead / (double)npart);
    // std::cout << "Dead < npart:" << dead << " " << npart
  }

  timer(
    [&]() {
      Kokkos::parallel_for(
        "push_p",
        npart,
        Push_kernel(ngh, coeff, fld, i1, i2, dx1, dx2, ux1, ux2, ux3, tag, nx1, nx2));
    },
    "push_p",
    10)();

  timer(
    [&]() {
      Kokkos::parallel_for(
        "kill_p",
        npart,
        KOKKOS_LAMBDA(index_t p) {
          RandomGenerator_t rand_gen = random_pool.get_state();
          if (rand_gen.frand() < 0.05) {
            tag(p) = tag::dead;
          }
          random_pool.free_state(rand_gen);
        });
    },
    "kill_p")();

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
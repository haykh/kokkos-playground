#include "playground.hpp"

#include <Kokkos_Random.hpp>

#include <fstream>
#include <iostream>

using RandomNumberPool_t = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
using RandomGenerator_t = typename RandomNumberPool_t::generator_type;
inline constexpr std::uint64_t RandomSeed = 0x123456789abcdef0;

using vec_t = float[3];

struct Maxwellian {
  Maxwellian(RandomNumberPool_t& random_pool) : pool { random_pool } {}

  // Juttner-Synge distribution
  KOKKOS_INLINE_FUNCTION void JS(vec_t& v, const float& temp) const {
    auto  rand_gen = pool.get_state();
    float u { ZERO }, eta { ZERO }, theta { ZERO };
    float X1 { ZERO }, X2 { ZERO };
    if (temp < 0.5) {
      // Juttner-Synge distribution using the Box-Muller method - non-relativistic

      u = rand_gen.frand();
      while (AlmostEqual(u, ZERO)) {
        u = rand_gen.frand();
      }
      eta   = math::sqrt(-TWO * math::log(u));
      theta = constant::TWO_PI * rand_gen.frand();
      while (AlmostEqual(theta, ZERO)) {
        theta = constant::TWO_PI * rand_gen.frand();
      }
      v[0] = eta * math::cos(theta) * math::sqrt(temp);
      v[1] = eta * math::sin(theta) * math::sqrt(temp);
      u    = rand_gen.frand();
      while (AlmostEqual(u, ZERO)) {
        u = rand_gen.frand();
      }
      eta   = math::sqrt(-TWO * math::log(u));
      theta = constant::TWO_PI * rand_gen.frand();
      while (AlmostEqual(theta, ZERO)) {
        theta = constant::TWO_PI * rand_gen.frand();
      }
      v[2] = eta * math::cos(theta) * math::sqrt(temp);

    } else {
      // Juttner-Synge distribution using the Sobol method - relativistic
      u = ONE;
      while (SQR(eta) <= SQR(u) + ONE) {
        while (AlmostEqual(X1, ZERO)) {
          X1 = rand_gen.frand() * rand_gen.frand() * rand_gen.frand();
        }
        u  = -temp * math::log(X1);
        X2 = rand_gen.frand();
        while (AlmostEqual(X2, 0)) {
          X2 = rand_gen.frand();
        }
        eta = -temp * math::log(X1 * X2);
      }
      X1   = rand_gen.frand();
      X2   = rand_gen.frand();
      v[0] = u * (TWO * X1 - ONE);
      v[2] = TWO * u * math::sqrt(X1 * (ONE - X1));
      v[1] = v[2] * math::cos(constant::TWO_PI * X2);
      v[2] = v[2] * math::sin(constant::TWO_PI * X2);
    }
    pool.free_state(rand_gen);
  }

  KOKKOS_INLINE_FUNCTION void operator()(vec_t& v, const float& temp) const {
    if (AlmostEqual(temp, ZERO)) {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    } else {
      JS(v, temp);
    }
  }

private:
  RandomNumberPool_t pool;
};

auto Playground() -> void {
  RandomNumberPool_t   pool { RandomSeed };
  const std::size_t    N { 100000 };
  Kokkos::View<float*> x("x", N);
  Kokkos::View<float*> u1("u1", N);
  Kokkos::View<float*> u2("u2", N);
  Kokkos::View<float*> u3("u3", N);
  const float          xmin { -10.0 }, xmax { 10.0 };
  const float          temp { 0.1 };

  Maxwellian maxwellian { pool };

  auto file = std::ofstream("dep.csv");

  Kokkos::parallel_for(
    "Playground",
    N,
    KOKKOS_LAMBDA(const std::size_t i) {
      RandomGenerator_t generator = pool.get_state();
      x(i)                        = xmin + (xmax - xmin) * generator.frand();
      vec_t ux;
      maxwellian(ux, temp);
      u1(i) = ux[0];
      u2(i) = ux[1];
      u3(i) = ux[2];
      pool.free_state(generator);
    });
  auto x_h  = Kokkos::create_mirror_view(x);
  auto u1_h = Kokkos::create_mirror_view(u1);
  auto u2_h = Kokkos::create_mirror_view(u2);
  auto u3_h = Kokkos::create_mirror_view(u3);

  Kokkos::deep_copy(x_h, x);
  Kokkos::deep_copy(u1_h, u1);
  Kokkos::deep_copy(u2_h, u2);
  Kokkos::deep_copy(u3_h, u3);
  for (auto i { 0 }; i < N; ++i) {
    file << x_h(i) << "," << u1_h(i) << "," << u2_h(i) << "," << u3_h(i)
         << std::endl;
  }
  file.close();
}

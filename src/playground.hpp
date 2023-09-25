#ifndef PLAYGROUND_HPP
#define PLAYGROUND_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#define SIGNf(x) (((x) < 0.0f) ? -1.0f : 1.0f)

namespace math = Kokkos;
using real_t   = float;
using index_t  = const std::size_t;
using RandomNumberPool_t = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
using RandomGenerator_t = typename RandomNumberPool_t::generator_type;
inline constexpr std::uint64_t RandomSeed = 0x123456789abcdef0;

inline constexpr float ONE   = 1.0f;
inline constexpr float TWO   = 2.0f;
inline constexpr float THREE = 3.0f;
inline constexpr float FOUR  = 4.0f;
inline constexpr float FIVE  = 5.0f;
inline constexpr float ZERO  = 0.0f;
inline constexpr float HALF  = 0.5f;
inline constexpr float INV_2 = 0.5f;
inline constexpr float INV_4 = 0.25f;

enum tag : short {
  dead  = 0,
  alive = 1,
};

enum em : int {
  ex1 = 0,
  ex2 = 1,
  ex3 = 2,
  bx1 = 3,
  bx2 = 4,
  bx3 = 5,
};

enum cur : int {
  jx1 = 0,
  jx2 = 1,
  jx3 = 2,
};

class InitFlds_kernel {

  Kokkos::View<real_t** [6]> fld;
  RandomNumberPool_t         random_pool;

public:
  InitFlds_kernel(Kokkos::View<real_t** [6]> fld, RandomNumberPool_t random_pool) :
    fld { fld },
    random_pool { random_pool } {}

  KOKKOS_INLINE_FUNCTION
  auto operator()(index_t i1, index_t i2) const -> void {
    RandomGenerator_t rand_gen = random_pool.get_state();
#pragma unroll
    for (auto& c : { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
      fld(i1, i2, c) = rand_gen.frand(-1.0f, 1.0f);
    }
    random_pool.free_state(rand_gen);
  }
};

class InitPrtls_kernel {
  Kokkos::View<int*>    i1;
  Kokkos::View<int*>    i2;
  Kokkos::View<float*>  dx1;
  Kokkos::View<float*>  dx2;
  Kokkos::View<real_t*> ux1;
  Kokkos::View<real_t*> ux2;
  Kokkos::View<real_t*> ux3;
  Kokkos::View<short*>  tag;

  const std::size_t  nx1;
  const std::size_t  nx2;
  RandomNumberPool_t random_pool;

public:
  InitPrtls_kernel(Kokkos::View<int*>    i1,
                   Kokkos::View<int*>    i2,
                   Kokkos::View<float*>  dx1,
                   Kokkos::View<float*>  dx2,
                   Kokkos::View<real_t*> ux1,
                   Kokkos::View<real_t*> ux2,
                   Kokkos::View<real_t*> ux3,
                   Kokkos::View<short*>  tag,
                   std::size_t           nx1,
                   std::size_t           nx2,
                   RandomNumberPool_t    random_pool) :
    i1 { i1 },
    i2 { i2 },
    dx1 { dx1 },
    dx2 { dx2 },
    ux1 { ux1 },
    ux2 { ux2 },
    ux3 { ux3 },
    tag { tag },
    nx1 { nx1 },
    nx2 { nx2 },
    random_pool { RandomSeed } {}

  KOKKOS_INLINE_FUNCTION
  auto operator()(index_t p) const -> void {
    RandomGenerator_t rand_gen = random_pool.get_state();
    i1(p)                      = rand_gen.rand(0, static_cast<int>(nx1));
    i2(p)                      = rand_gen.rand(0, static_cast<int>(nx2));
    dx1(p)                     = rand_gen.frand(0.0f, 1.0f);
    dx2(p)                     = rand_gen.frand(0.0f, 1.1f);
    ux1(p)                     = rand_gen.frand(-1.0f, 1.0f);
    ux2(p)                     = rand_gen.frand(-1.0f, 1.0f);
    ux3(p)                     = rand_gen.frand(-1.0f, 1.0f);
    tag(p)                     = tag::alive;
    random_pool.free_state(rand_gen);
  }
};

class Push_kernel {
  const std::size_t ngh;
  const real_t      coeff;

  Kokkos::View<real_t** [6]> fld;
  Kokkos::View<int*>         i1;
  Kokkos::View<int*>         i2;
  Kokkos::View<float*>       dx1;
  Kokkos::View<float*>       dx2;
  Kokkos::View<real_t*>      ux1;
  Kokkos::View<real_t*>      ux2;
  Kokkos::View<real_t*>      ux3;
  Kokkos::View<short*>       tag;

  const int nx1;
  const int nx2;

public:
  Push_kernel(std::size_t                ngh,
              real_t                     coeff,
              Kokkos::View<real_t** [6]> fld,
              Kokkos::View<int*>         i1,
              Kokkos::View<int*>         i2,
              Kokkos::View<float*>       dx1,
              Kokkos::View<float*>       dx2,
              Kokkos::View<real_t*>      ux1,
              Kokkos::View<real_t*>      ux2,
              Kokkos::View<real_t*>      ux3,
              Kokkos::View<short*>       tag,
              std::size_t                nx1,
              std::size_t                nx2) :
    ngh { ngh },
    coeff { coeff },
    fld { fld },
    i1 { i1 },
    i2 { i2 },
    dx1 { dx1 },
    dx2 { dx2 },
    ux1 { ux1 },
    ux2 { ux2 },
    ux3 { ux3 },
    tag { tag },
    nx1 { (int)nx1 },
    nx2 { (int)nx2 } {}

  KOKKOS_INLINE_FUNCTION
  auto operator()(index_t p) const -> void {
    if (tag(p) == static_cast<short>(tag::alive)) {
      {
        const auto i1_ = i1(p) + static_cast<int>(ngh);
        const auto i2_ = i2(p) + static_cast<int>(ngh);
        real_t     c000, c100, c010, c110;

        c000 = HALF * (fld(i1_, i2_, em::ex1) + fld(i1_ - 1, i2_, em::ex1));
        c100 = HALF * (fld(i1_, i2_, em::ex1) + fld(i1_ + 1, i2_, em::ex1));
        c010 = HALF *
               (fld(i1_, i2_ + 1, em::ex1) + fld(i1_ - 1, i2_ + 1, em::ex1));
        c110 = HALF *
               (fld(i1_, i2_ + 1, em::ex1) + fld(i1_ + 1, i2_ + 1, em::ex1));
        const auto ex1_int = (c000 * (ONE - dx1(p)) + c100 * dx1(p)) *
                               (ONE - dx2(p)) +
                             (c010 * (ONE - dx1(p)) + c110 * dx1(p)) * dx2(p);

        c000 = HALF * (fld(i1_, i2_, em::ex2) + fld(i1_, i2_ - 1, em::ex2));
        c100 = HALF *
               (fld(i1_ + 1, i2_, em::ex2) + fld(i1_ + 1, i2_ - 1, em::ex2));
        c010 = HALF * (fld(i1_, i2_, em::ex2) + fld(i1_, i2_ + 1, em::ex2));
        c110 = HALF *
               (fld(i1_ + 1, i2_, em::ex2) + fld(i1_ + 1, i2_ + 1, em::ex2));
        const auto ex2_int = (c000 * (ONE - dx1(p)) + c100 * dx1(p)) *
                               (ONE - dx2(p)) +
                             (c010 * (ONE - dx1(p)) + c110 * dx1(p)) * dx2(p);

        c000               = fld(i1_, i2_, em::ex3);
        c100               = fld(i1_ + 1, i2_, em::ex3);
        c010               = fld(i1_, i2_ + 1, em::ex3);
        c110               = fld(i1_ + 1, i2_ + 1, em::ex3);
        const auto ex3_int = (c000 * (ONE - dx1(p)) + c100 * dx1(p)) *
                               (ONE - dx2(p)) +
                             (c010 * (ONE - dx1(p)) + c110 * dx1(p)) * dx2(p);

        c000 = HALF * (fld(i1_, i2_, em::bx1) + fld(i1_, i2_ - 1, em::bx1));
        c100 = HALF *
               (fld(i1_ + 1, i2_, em::bx1) + fld(i1_ + 1, i2_ - 1, em::bx1));
        c010 = HALF * (fld(i1_, i2_, em::bx1) + fld(i1_, i2_ + 1, em::bx1));
        c110 = HALF *
               (fld(i1_ + 1, i2_, em::bx1) + fld(i1_ + 1, i2_ + 1, em::bx1));
        const auto bx1_int = (c000 * (ONE - dx1(p)) + c100 * dx1(p)) *
                               (ONE - dx2(p)) +
                             (c010 * (ONE - dx1(p)) + c110 * dx1(p)) * dx2(p);

        c000 = HALF * (fld(i1_ - 1, i2_, em::bx2) + fld(i1_, i2_, em::bx2));
        c100 = HALF * (fld(i1_, i2_, em::bx2) + fld(i1_ + 1, i2_, em::bx2));
        c010 = HALF *
               (fld(i1_ - 1, i2_ + 1, em::bx2) + fld(i1_, i2_ + 1, em::bx2));
        c110 = HALF *
               (fld(i1_, i2_ + 1, em::bx2) + fld(i1_ + 1, i2_ + 1, em::bx2));
        const auto bx2_int = (c000 * (ONE - dx1(p)) + c100 * dx1(p)) *
                               (ONE - dx2(p)) +
                             (c010 * (ONE - dx1(p)) + c110 * dx1(p)) * dx2(p);

        c000 = INV_4 *
               (fld(i1_ - 1, i2_ - 1, em::bx3) + fld(i1_ - 1, i2_, em::bx3) +
                fld(i1_, i2_ - 1, em::bx3) + fld(i1_, i2_, em::bx3));
        c100 = INV_4 *
               (fld(i1_, i2_ - 1, em::bx3) + fld(i1_, i2_, em::bx3) +
                fld(i1_ + 1, i2_ - 1, em::bx3) + fld(i1_ + 1, i2_, em::bx3));
        c010 = INV_4 *
               (fld(i1_ - 1, i2_, em::bx3) + fld(i1_ - 1, i2_ + 1, em::bx3) +
                fld(i1_, i2_, em::bx3) + fld(i1_, i2_ + 1, em::bx3));
        c110 = INV_4 *
               (fld(i1_, i2_, em::bx3) + fld(i1_, i2_ + 1, em::bx3) +
                fld(i1_ + 1, i2_, em::bx3) + fld(i1_ + 1, i2_ + 1, em::bx3));
        const auto bx3_int = (c000 * (ONE - dx1(p)) + c100 * dx1(p)) *
                               (ONE - dx2(p)) +
                             (c010 * (ONE - dx1(p)) + c110 * dx1(p)) * dx2(p);

        ux1(p) += coeff * (ex1_int + ux2(p) * bx3_int - ux3(p) * bx2_int);
        ux2(p) += coeff * (ex2_int + ux3(p) * bx1_int - ux1(p) * bx3_int);
        ux3(p) += coeff * (ex3_int + ux1(p) * bx2_int - ux2(p) * bx1_int);
      }
      const auto inv_gamma = ONE / math::sqrt(ONE + ux1(p) * ux1(p) +
                                              ux2(p) * ux2(p) + ux3(p) * ux3(p));
      {
        dx1(p) += static_cast<float>(coeff * ux1(p) * inv_gamma);
        auto temp_i { static_cast<int>(dx1(p)) };
        auto temp_r { math::fmax(SIGNf(dx1(p)) + temp_i, static_cast<float>(temp_i)) -
                      static_cast<float>(1.0) };
        temp_i = static_cast<int>(temp_r);
        i1(p)  = (i1(p) + temp_i + nx1) % nx1;
        dx1(p) = dx1(p) - temp_r;
      }

      {
        dx2(p)      += static_cast<float>(coeff * ux2(p) * inv_gamma);
        auto temp_i  = static_cast<int>(dx2(p));
        auto temp_r  = math::fmax(SIGNf(dx2(p)) + temp_i,
                                 static_cast<float>(temp_i)) -
                      static_cast<float>(1.0);
        temp_i = static_cast<int>(temp_r);
        i2(p)  = (i2(p) + temp_i + nx2) % nx2;
        dx2(p) = dx2(p) - temp_r;
      }
      if (i1(p) >= nx1 || i1(p) < 0) {
        throw std::runtime_error("ERROR i1" + std::to_string(i1(p)));
      }
      if (i2(p) >= nx2 || i2(p) < 0) {
        throw std::runtime_error("ERROR i2" + std::to_string(i2(p)));
      }
    }
  }
};

auto Playground() -> void;

#endif // PLAYGROUND_HPP
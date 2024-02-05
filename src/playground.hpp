#ifndef PLAYGROUND_HPP
#define PLAYGROUND_HPP

#include <Kokkos_Core.hpp>

#include <cstdint>

#define SQR(x) ((x) * (x))

namespace math = Kokkos;

inline constexpr float ONE    = 1.0f;
inline constexpr float TWO    = 2.0f;
inline constexpr float THREE  = 3.0f;
inline constexpr float FOUR   = 4.0f;
inline constexpr float ZERO   = 0.0f;
inline constexpr float HALF   = 0.5f;
inline constexpr float INV_2  = 0.5f;
inline constexpr float INV_4  = 0.25f;
inline constexpr float INV_8  = 0.125f;
inline constexpr float INV_16 = 0.0625f;
inline constexpr float INV_32 = 0.03125f;
inline constexpr float INV_64 = 0.015625f;

namespace constant {
  inline constexpr float HALF_PI    = 1.57079632679489661923;
  inline constexpr float PI         = 3.14159265358979323846;
  inline constexpr float INV_PI     = 0.31830988618379067154;
  inline constexpr float PI_SQR     = 9.86960440108935861882;
  inline constexpr float INV_PI_SQR = 0.10132118364233777144;
  inline constexpr float TWO_PI     = 6.28318530717958647692;
  inline constexpr float E          = 2.71828182845904523536;
  inline constexpr float SQRT2      = 1.41421356237309504880;
  inline constexpr float INV_SQRT2  = 0.70710678118654752440;
  inline constexpr float SQRT3      = 1.73205080756887729352;
} // namespace constant

KOKKOS_INLINE_FUNCTION
bool AlmostEqual(float a, float b, float epsilon = 0.00001f) {
  if (a == b) {
    return true;
  } else {
    float diff { math::abs(a - b) };
    if (diff <= 1e-6) {
      return true;
    } else {
      return diff <= math::min(math::abs(a), math::abs(b)) * epsilon;
    }
  }
}

auto Playground() -> void;

#endif // PLAYGROUND_HPP

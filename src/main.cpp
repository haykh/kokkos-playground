#include <Kokkos_Core.hpp>

#include <iostream>

struct MetricBase {
  KOKKOS_INLINE_FUNCTION
  virtual auto hij() const -> float = 0;

  KOKKOS_INLINE_FUNCTION
  virtual void convert(const float&, float&) const = 0;
};

struct MetricSpherical : public MetricBase {
  KOKKOS_INLINE_FUNCTION
  void convert(const float& a, float& b) const override {
    b = a * this->hij();
  };
};

struct MyMetric : public MetricSpherical {
  KOKKOS_INLINE_FUNCTION
  auto hij() const -> float override {
    return 1.23;
  };
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    MyMetric m;
    Kokkos::parallel_for(
      "test", 10, KOKKOS_LAMBDA(const int& i) {
        float a = 1.0;
        float b = 0.0;
        m.convert(a, b);
        printf("a = %f, b = %f\n", a, b);
      });
  }
  Kokkos::finalize();
  return 0;
}

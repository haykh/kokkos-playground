#ifndef TIMER_H
#define TIMER_H

#include <Kokkos_Core.hpp>

#include <chrono>
#include <iostream>
#include <string>

template <typename F>
class TimerWrapper {
  F           func;
  std::string name;
  int         ntimes;

public:
  TimerWrapper(F func, std::string name, int ntimes) :
    func { func },
    name { name },
    ntimes { ntimes } {}

  template <typename... Args>
  auto operator()(Args... args) {
    unsigned long runtime { 0 };
    long double   avg_runtime { 0.0 };

    struct EndScope {
      std::string& name;
      long double& time;

      EndScope(std::string& name, long double& time) :
        name { name },
        time { time } {}

      ~EndScope() {
        std::cout << name << " : OK, avg. runtime: " << time << " ms\n"
                  << std::endl;
      }
    } endScope(name, avg_runtime);

    std::cout << "executing: " << name << " " << ntimes << " times" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i { 0 }; i < ntimes; ++i) {
      func(args...);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    avg_runtime = static_cast<long double>(runtime) /
                  static_cast<long double>(ntimes);
  }
};

template <typename F>
auto timer(F func, std::string name, int ntimes = 1) {
  return TimerWrapper<F>(func, name, ntimes);
}

#endif // TIMER_H
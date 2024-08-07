#include "playground.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#ifdef CUDA_ENABLED
  #include <thrust/execution_policy.h>
  #include <thrust/iterator/zip_iterator.h>
  #include <thrust/sort.h>
#endif

#include <chrono>
#include <iostream>

auto Playground() -> void {
  RunOnce([]() {
#ifdef MPI_ENABLED
    std::cout << "MPI: ON\n";
#else
    std::cout << "MPI: OFF\n";
#endif

#ifdef CUDA_ENABLED
    std::cout << "CUDA: ON\n";
#else
    std::cout << "CUDA: OFF\n";
#endif
  });

  constexpr auto x1min_g = 0.0f, x1max_g = 1.0f;
  constexpr auto x2min_g = -1.0f, x2max_g = 1.0f;
  constexpr auto Nx1_g = 2560, Nx2_g = 1280;
  constexpr auto Nmax    = static_cast<std::size_t>(1e8);
  constexpr auto nrepeat = 25;
#ifdef MPI_ENABLED
  int ncpu_x1 = 2, ncpu_x2 = 2;
#endif

  Kokkos::View<float*> dx1("dx1", Nmax);
  Kokkos::View<float*> dx1_prev("dx1_prev", Nmax);
  Kokkos::View<float*> dx2("dx2", Nmax);
  Kokkos::View<float*> dx2_prev("dx2_prev", Nmax);
  Kokkos::View<int*>   i1("i1", Nmax);
  Kokkos::View<int*>   i1_prev("i1_prev", Nmax);
  Kokkos::View<int*>   i2("i2", Nmax);
  Kokkos::View<int*>   i2_prev("i2_prev", Nmax);
  Kokkos::View<float*> ux1("ux1", Nmax);
  Kokkos::View<float*> ux2("ux2", Nmax);
  Kokkos::View<float*> ux3("ux3", Nmax);
  Kokkos::View<short*> tag("tag", Nmax);

  RandomNumberPool_t pool { RandomSeed };
  std::size_t        Nactive = static_cast<std::size_t>(1e7);

  float x1min, x1max, x2min, x2max;
  int   Nx1, Nx2;
#ifdef MPI_ENABLED
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  Nx1   = Nx1_g / ncpu_x1;
  Nx2   = Nx2_g / ncpu_x2;
  x1min = x1min_g + (x1max_g - x1min_g) * (mpi_rank % ncpu_x1) / ncpu_x1;
  x1max = x1min_g + (x1max_g - x1min_g) * ((mpi_rank % ncpu_x1) + 1) / ncpu_x1;
  x2min = x2min_g + (x2max_g - x2min_g) * (mpi_rank / ncpu_x1) / ncpu_x2;
  x2max = x2min_g + (x2max_g - x2min_g) * ((mpi_rank / ncpu_x1) + 1) / ncpu_x2;
#else
  Nx1   = Nx1_g;
  Nx2   = Nx2_g;
  x1min = x1min_g;
  x1max = x1max_g;
  x2min = x2min_g;
  x2max = x2max_g;
#endif

  std::vector<std::size_t> durations;
  for (auto r = 0; r < nrepeat; ++r) {
    // initialize coordinates & velocities
    Init(Nactive, Nx1, Nx2, dx1, dx2, i1, i2, ux1, ux2, ux3, tag, pool);

    // push particles & apply boundary conditions
    Push(Nactive,
         Nx1,
         Nx2,
         x1min,
         x1max,
         x2min,
         x2max,
         dx1,
         dx1_prev,
         dx2,
         dx2_prev,
         i1,
         i1_prev,
         i2,
         i2_prev,
         ux1,
         ux2,
         tag,
         pool);

    // count alive, send, and dead particles
    auto count   = Count(Nactive, tag);
    auto n_alive = count[Tag::Alive];
    auto n_dead  = count[Tag::Dead];
    auto n_send  = (std::size_t)0;
    for (auto t = Tag::Alive + 1; t < Tag::Ntags; ++t) {
      n_send += count[t];
    }

    if (CheckSorted(tag, n_alive, n_send, n_dead)) {
      throw std::runtime_error("Sorted before sorting");
    }

    auto duration = Sort(Nactive,
                         dx1,
                         dx1_prev,
                         dx2,
                         dx2_prev,
                         i1,
                         i1_prev,
                         i2,
                         i2_prev,
                         ux1,
                         ux2,
                         ux3,
                         tag);
    if (r == nrepeat - 1) {
      PrintTags(Nactive, tag);
    }
    durations.push_back(duration);

    if (!CheckSorted(tag, n_alive, n_send, n_dead)) {
      throw std::runtime_error("Not sorted after sorting");
    }
  }

  RunSequentially(
    [](auto&& durations) {
      std::cout << "Average sort time with Kokkos: "
                << static_cast<float>(
                     std::accumulate(durations.begin(), durations.end(), 0)) /
                     static_cast<float>(durations.size())
                << " ms\n";
    },
    durations);

#ifdef CUDA_ENABLED
  for (auto r = 0; r < nrepeat; ++r) {
    // initialize coordinates & velocities
    Init(Nactive, Nx1, Nx2, dx1, dx2, i1, i2, ux1, ux2, ux3, tag, pool);

    // push particles & apply boundary conditions
    Push(Nactive,
         Nx1,
         Nx2,
         x1min,
         x1max,
         x2min,
         x2max,
         dx1,
         dx1_prev,
         dx2,
         dx2_prev,
         i1,
         i1_prev,
         i2,
         i2_prev,
         ux1,
         ux2,
         tag,
         pool);

    // count alive, send, and dead particles
    auto count   = Count(Nactive, tag);
    auto n_alive = count[Tag::Alive];
    auto n_dead  = count[Tag::Dead];
    auto n_send  = (std::size_t)0;
    for (auto t = Tag::Alive + 1; t < Tag::Ntags; ++t) {
      n_send += count[t];
    }

    if (CheckSorted(tag, n_alive, n_send, n_dead)) {
      throw std::runtime_error("Sorted before sorting");
    }

    auto duration = SortWithThrust(Nactive,
                                   dx1,
                                   dx1_prev,
                                   dx2,
                                   dx2_prev,
                                   i1,
                                   i1_prev,
                                   i2,
                                   i2_prev,
                                   ux1,
                                   ux2,
                                   ux3,
                                   tag);
    if (r == nrepeat - 1) {
      PrintTags(Nactive, tag);
    }
    durations.push_back(duration);

    if (!CheckSorted(tag, n_alive, n_send, n_dead)) {
      throw std::runtime_error("Not sorted after sorting");
    }
  }

  RunSequentially(
    [](auto&& durations) {
      std::cout << "Average sort time with thrust: "
                << static_cast<float>(
                     std::accumulate(durations.begin(), durations.end(), 0)) /
                     static_cast<float>(durations.size())
                << " ms\n";
    },
    durations);
#endif
}

auto Init(std::size_t          nactive,
          int                  Nx1,
          int                  Nx2,
          Kokkos::View<float*> dx1,
          Kokkos::View<float*> dx2,
          Kokkos::View<int*>   i1,
          Kokkos::View<int*>   i2,
          Kokkos::View<float*> ux1,
          Kokkos::View<float*> ux2,
          Kokkos::View<float*> ux3,
          Kokkos::View<short*> tag,
          RandomNumberPool_t&  pool) -> void {
  Kokkos::parallel_for(
    "Init",
    nactive,
    KOKKOS_LAMBDA(const std::size_t p) {
      auto generator = pool.get_state();
      dx1(p)         = generator.frand();
      dx2(p)         = generator.frand();
      i1(p)          = static_cast<int>(generator.frand() * Nx1);
      i2(p)          = static_cast<int>(generator.frand() * Nx2);
      ux1(p)         = 0.1f * 2.0f * (generator.frand() - 0.5f);
      ux2(p)         = 0.2f * 2.0f * (generator.frand() - 0.5f);
      ux3(p)         = 0.1f * 2.0f * (generator.frand() - 0.5f);
      tag(p)         = Tag::Alive;
      pool.free_state(generator);
    });
}

auto Push(std::size_t          nactive,
          int                  Nx1,
          int                  Nx2,
          float                x1min,
          float                x1max,
          float                x2min,
          float                x2max,
          Kokkos::View<float*> dx1,
          Kokkos::View<float*> dx1_prev,
          Kokkos::View<float*> dx2,
          Kokkos::View<float*> dx2_prev,
          Kokkos::View<int*>   i1,
          Kokkos::View<int*>   i1_prev,
          Kokkos::View<int*>   i2,
          Kokkos::View<int*>   i2_prev,
          Kokkos::View<float*> ux1,
          Kokkos::View<float*> ux2,
          Kokkos::View<short*> tag,
          RandomNumberPool_t&  pool) -> void {
  const auto delta_x1 = (x1max - x1min) / (float)Nx1;
  const auto delta_x2 = (x2max - x2min) / (float)Nx2;
  Kokkos::parallel_for(
    "Push",
    nactive,
    KOKKOS_LAMBDA(const std::size_t p) {
      auto generator  = pool.get_state();
      auto x1         = (dx1(p) + static_cast<float>(i1(p))) * delta_x1 + x1min;
      auto x2         = (dx2(p) + static_cast<float>(i2(p))) * delta_x2 + x2min;
      x1             += ux1(p);
      x2             += ux2(p);
      i1_prev(p)      = i1(p);
      i2_prev(p)      = i2(p);
      dx1_prev(p)     = dx1(p);
      dx2_prev(p)     = dx2(p);
      i1(p)           = static_cast<int>((x1 - x1min) / delta_x1);
      dx1(p)          = (x1 - x1min) / delta_x1 - static_cast<float>(i1(p));
      i2(p)           = static_cast<int>((x2 - x2min) / delta_x2);
      dx2(p)          = (x2 - x2min) / delta_x2 - static_cast<float>(i2(p));
      if (i1(p) < 0) {
        if (i2(p) < 0) {
          tag(p) = Tag::Send_MM;
        } else if (i2(p) >= Nx2) {
          tag(p) = Tag::Send_MP;
        } else {
          tag(p) = Tag::Send_M0;
        }
      } else if (i1(p) >= Nx1) {
        if (i2(p) < 0) {
          tag(p) = Tag::Send_PM;
        } else if (i2(p) >= Nx2) {
          tag(p) = Tag::Send_PP;
        } else {
          tag(p) = Tag::Send_P0;
        }
      } else if (i2(p) < 0) {
        tag(p) = Tag::Send_0M;
      } else if (i2(p) >= Nx2) {
        tag(p) = Tag::Send_0P;
      } else if (generator.frand() < 0.01f) {
        tag(p) = Tag::Dead;
      }
      pool.free_state(generator);
    });
}

auto Count(std::size_t nactive, Kokkos::View<short*> tag)
  -> std::vector<std::size_t> {
  Kokkos::View<std::size_t*> count("count", Tag::Ntags);
  Kokkos::parallel_for(
    "Count",
    nactive,
    KOKKOS_LAMBDA(const std::size_t p) {
      Kokkos::atomic_increment(&count(tag(p)));
    });
  auto count_h = Kokkos::create_mirror_view(count);
  Kokkos::deep_copy(count_h, count);
  std::vector<std::size_t> count_v(count_h.data(), count_h.data() + Tag::Ntags);
  return count_v;
}

auto Sort(std::size_t          nactive,
          Kokkos::View<float*> dx1,
          Kokkos::View<float*> dx1_prev,
          Kokkos::View<float*> dx2,
          Kokkos::View<float*> dx2_prev,
          Kokkos::View<int*>   i1,
          Kokkos::View<int*>   i1_prev,
          Kokkos::View<int*>   i2,
          Kokkos::View<int*>   i2_prev,
          Kokkos::View<float*> ux1,
          Kokkos::View<float*> ux2,
          Kokkos::View<float*> ux3,
          Kokkos::View<short*> tag) -> std::size_t {
  using KeyType = Kokkos::View<short*>;
  using BinOp   = BinTag<KeyType>;
  BinOp bin_op(Tag::Ntags);
  auto  slice = std::pair<std::size_t, std::size_t> { 0, nactive };

  Fence();
  auto tbegin = std::chrono::high_resolution_clock::now();
  Kokkos::BinSort<KeyType, BinOp> Sorter(Kokkos::subview(tag, slice), bin_op, false);
  Sorter.create_permute_vector();
  Sorter.sort(Kokkos::subview(dx1, slice));
  Sorter.sort(Kokkos::subview(dx1_prev, slice));
  Sorter.sort(Kokkos::subview(dx2, slice));
  Sorter.sort(Kokkos::subview(dx2_prev, slice));
  Sorter.sort(Kokkos::subview(i1, slice));
  Sorter.sort(Kokkos::subview(i1_prev, slice));
  Sorter.sort(Kokkos::subview(i2, slice));
  Sorter.sort(Kokkos::subview(i2_prev, slice));
  Sorter.sort(Kokkos::subview(ux1, slice));
  Sorter.sort(Kokkos::subview(ux2, slice));
  Sorter.sort(Kokkos::subview(ux3, slice));
  Sorter.sort(Kokkos::subview(tag, slice));
  auto tend = std::chrono::high_resolution_clock::now();
  Fence();

  return std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();
}

#ifdef CUDA_ENABLED
auto SortWithThrust(std::size_t          nactive,
                    Kokkos::View<float*> dx1,
                    Kokkos::View<float*> dx1_prev,
                    Kokkos::View<float*> dx2,
                    Kokkos::View<float*> dx2_prev,
                    Kokkos::View<int*>   i1,
                    Kokkos::View<int*>   i1_prev,
                    Kokkos::View<int*>   i2,
                    Kokkos::View<int*>   i2_prev,
                    Kokkos::View<float*> ux1,
                    Kokkos::View<float*> ux2,
                    Kokkos::View<float*> ux3,
                    Kokkos::View<short*> tag) -> std::size_t {
  Fence();
  auto tbegin = std::chrono::high_resolution_clock::now();
  auto zipit  = thrust::make_zip_iterator(thrust::make_tuple(dx1.data(),
                                                            dx1_prev.data(),
                                                            dx2.data(),
                                                            dx2_prev.data(),
                                                            i1.data(),
                                                            i1_prev.data(),
                                                            i2.data(),
                                                            i2_prev.data(),
                                                            ux1.data(),
                                                            ux2.data(),
                                                            ux3.data(),
                                                            tag.data()));
  thrust::sort_by_key(thrust::device,
                      tag.data(),
                      tag.data() + nactive,
                      zipit,
                      TagComparator());
  auto tend = std::chrono::high_resolution_clock::now();
  Fence();
  return std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();
}
#endif

auto Fence() -> void {
  Kokkos::fence();
#ifdef MPI_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

auto CheckSorted(Kokkos::View<short*> tag,
                 std::size_t          nalive,
                 std::size_t          nsend,
                 std::size_t          ndead) -> bool {
  Kokkos::View<std::size_t> incorrect("count");
  Kokkos::parallel_for(
    "Count",
    nalive + nsend + ndead,
    KOKKOS_LAMBDA(const std::size_t p) {
      if (((tag(p) != Tag::Alive) && (p < nalive)) ||
          (((tag(p) == Tag::Alive) || (tag(p) == Tag::Dead)) &&
           (p < nalive + nsend) && (p >= nalive)) ||
          ((tag(p) != Tag::Dead) && (p >= nalive + nsend))) {
        Kokkos::atomic_increment(&incorrect());
      }
    });
  auto incorrect_h = Kokkos::create_mirror_view(incorrect);
  Kokkos::deep_copy(incorrect_h, incorrect);
  return incorrect_h() == 0;
}

auto PrintTags(std::size_t nactive, Kokkos::View<short*> tag) -> void {
  auto tag_h = Kokkos::create_mirror_view(tag);
  Kokkos::deep_copy(tag_h, tag);
  auto cur_tag = tag_h(0);
  auto nglob = 1, ncur = 1, nmax = 4;
  std::cout << "Tags: " << cur_tag << " ";
  for (auto p = 1; p < nactive; ++p) {
    nglob++;
    if (tag_h(p) == cur_tag) {
      ncur++;
      if (ncur > nmax) {
        continue;
      }
      std::cout << cur_tag << " ";
    } else {
      cur_tag = tag_h(p);
      if (ncur > nmax) {
        std::cout << "...(" << ncur << ")... ";
      }
      ncur = 1;
      std::cout << cur_tag << " ";
    }
  }
  if (ncur > nmax) {
    std::cout << "...(" << ncur << ")... ";
  }
  std::cout << "\n";
  std::cout << "Total: " << nglob << "\n";
}

auto StringizeTag(short tag) -> std::string {
  switch (tag) {
    case Tag::Dead:
      return "Dead";
    case Tag::Alive:
      return "Alive";
    case Tag::Send_M0:
      return "Send_M0";
    case Tag::Send_P0:
      return "Send_P0";
    case Tag::Send_0M:
      return "Send_0M";
    case Tag::Send_0P:
      return "Send_0P";
    case Tag::Send_MM:
      return "Send_MM";
    case Tag::Send_MP:
      return "Send_MP";
    case Tag::Send_PM:
      return "Send_PM";
    case Tag::Send_PP:
      return "Send_PP";
    default:
      return "Unknown";
  }
}

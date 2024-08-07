#ifndef PLAYGROUND_HPP
#define PLAYGROUND_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#ifdef CUDA_ENABLED
  #include <cuda_runtime.h>
#endif

#include <string>
#include <vector>

enum Tag : short {
  Dead = 0,
  Alive,
  Send_M0,
  Send_P0,
  Send_0M,
  Send_0P,
  Send_MM,
  Send_MP,
  Send_PM,
  Send_PP,
  Ntags,
};

auto StringizeTag(short tag) -> std::string;

using RandomNumberPool_t = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
using RandomGenerator_t = typename RandomNumberPool_t::generator_type;
inline constexpr std::uint64_t RandomSeed = 0x123456789abcdef0;

auto Playground() -> void;

auto Init(std::size_t,
          int,
          int,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<short*>,
          RandomNumberPool_t&) -> void;

auto Push(std::size_t,
          int,
          int,
          float,
          float,
          float,
          float,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<short*>,
          RandomNumberPool_t&) -> void;

auto Sort(std::size_t,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<short*>) -> std::size_t;

#ifdef CUDA_ENABLED
struct TagComparator {
  __host__ __device__ inline bool operator()(const short& t1, const short& t2) {
    return (t1 != 0) * ((t2 == 0) + (t1 < t2));
  }
};

auto SortWithThrust(std::size_t,
                    Kokkos::View<float*>,
                    Kokkos::View<float*>,
                    Kokkos::View<float*>,
                    Kokkos::View<float*>,
                    Kokkos::View<int*>,
                    Kokkos::View<int*>,
                    Kokkos::View<int*>,
                    Kokkos::View<int*>,
                    Kokkos::View<float*>,
                    Kokkos::View<float*>,
                    Kokkos::View<float*>,
                    Kokkos::View<short*>) -> std::size_t;
#endif

auto Count(std::size_t, Kokkos::View<short*>) -> std::vector<std::size_t>;

auto Fence() -> void;

auto CheckSorted(Kokkos::View<short*>, std::size_t, std::size_t, std::size_t)
  -> bool;

auto PrintTags(std::size_t, Kokkos::View<short*>) -> void;

template <class Function, class... Args>
auto RunSequentially(Function&& function, Args&&... args) -> void {
#ifdef MPI_ENABLED
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  for (auto rank = 0; rank < mpi_size; ++rank) {
    if (mpi_rank == rank) {
      function(std::forward<Args>(args)...);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#else
  function(std::forward<Args>(args)...);
#endif
}

template <class Function, class... Args>
auto RunOnce(Function&& function, Args&&... args) -> void {
#ifdef MPI_ENABLED
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    function(std::forward<Args>(args)...);
  }
#else
  function(std::forward<Args>(args)...);
#endif
}

template <class KeyViewType>
struct BinTag {
  BinTag(const int& max_bins) : m_max_bins { max_bins } {}

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION auto bin(ViewType& keys, const int& i) const -> int {
    return (keys(i) == 0) ? (Tag::Ntags - 1) : (keys(i) - 1);
  }

  KOKKOS_INLINE_FUNCTION auto max_bins() const -> int {
    return m_max_bins;
  }

  template <class ViewType, typename iT1, typename iT2>
  KOKKOS_INLINE_FUNCTION auto operator()(ViewType&, iT1&, iT2&) const -> bool {
    return false;
  }

private:
  const int m_max_bins;
};

#endif // PLAYGROUND_HPP
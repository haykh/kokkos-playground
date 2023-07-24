#include <Kokkos_Core.hpp>
#include <stdio.h>

#include <chrono>
#include <iostream>
#include <numeric>

#ifdef ADIOS2_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

namespace math = Kokkos;

auto Initialize(int argc, char* argv[]) -> void;
auto Finalize() -> void;

#ifdef ADIOS2_ENABLED
auto main(int argc, char* argv[]) -> int {
  Initialize(argc, argv);
  try {
    int mpi_rank = 0, mpi_size = 1;
#  ifndef MPI_ENABLED
    adios2::ADIOS adios;
#  else
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    adios2::ADIOS adios(MPI_COMM_WORLD);
#  endif

    auto io = adios.DeclareIO("Test-Output");
    io.SetEngine("HDF5");

    const auto maxptl = 100000;
    const auto nsteps = 10;

    auto       myarr1 = Kokkos::View<double*>("myarr1", maxptl);

    io.DefineVariable<double>("myarr1", {}, {}, { adios2::UnknownDim });

    auto writer       = io.Open("test.h5", adios2::Mode::Write);

    // adios.EnterComputationBlock();
    for (auto t { 0 }; t < nsteps; ++t) {
      if (mpi_rank == 0) {
        printf("Timestep #%d\n\n", t);
      }
      const auto nprt = (std::size_t)(100 * (t + 1) * (mpi_rank + 1));
      if (nprt > maxptl) {
        throw std::runtime_error("Too many particles");
      }
      Kokkos::parallel_for(
        "fill_myarr1", nprt, KOKKOS_LAMBDA(const int i) {
          myarr1(i) += static_cast<double>(i);
        });
      // adios.ExitComputationBlock();
      writer.BeginStep();

#  ifdef MPI_ENABLED
      std::vector<std::size_t> sizes_g(mpi_size), offsets(mpi_size);
      MPI_Allgather(&nprt, 1, MPI_DOUBLE, sizes_g.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
      const auto total_nprt = std::accumulate(sizes_g.begin(), sizes_g.end(),
      (std::size_t)0); offsets[0]            = 0; for (auto i { 1 }; i < sizes_g.size();
      ++i) {
        offsets[i] = offsets[i - 1] + sizes_g[i - 1];
      }
      const auto offset = offsets[mpi_rank];
#  else     // not MPI_ENABLED
      const auto offset     = (std::size_t)0;
      const auto total_nprt = nprt;
#  endif    // MPI_ENABLED

      auto myarr1_slice = Kokkos::View<double*>("myarr1_slice", nprt);
      auto slice        = std::make_pair((std::size_t)0, (std::size_t)nprt);
      Kokkos::deep_copy(myarr1_slice, Kokkos::subview(myarr1, slice));
      printf(
        "#%d: nprt = %d, offset = %d, total_nprt = %d\n", mpi_rank, nprt, offset,
        total_nprt);

      auto var = io.InquireVariable<double>("myarr1");
      var.SetSelection({ {}, { nprt } });
      writer.Put<double>(var, myarr1_slice, adios2::Mode::Sync);

      writer.EndStep();
      // adios.EnterComputationBlock();
      if (mpi_rank == 0) {
        printf("---\n\n");
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << "Closing file\n";
    writer.Close();

  } catch (std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << '\n';
    Finalize();
    return 1;
  }
  Finalize();
  return 0;
}
#else
auto main() -> int {
  std::cout << "ADIOS2 is not enabled.\n";
  return 0;
}
#endif

auto Initialize(int argc, char* argv[]) -> void {
  Kokkos::initialize(argc, argv);
#ifdef MPI_ENABLED
  MPI_Init(&argc, &argv);
#endif
}

auto Finalize() -> void {
#ifdef MPI_ENABLED
  MPI_Finalize();
#endif
  Kokkos::finalize();
}
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

    const std::size_t maxptl = 200;
    const std::size_t ncells = 100;
    const auto        nsteps = 10;

    auto              myarr1 = Kokkos::View<double*>("myarr1", maxptl);
    auto              myarr2 = Kokkos::View<double*>("myarr2", ncells);

    io.DefineVariable<double>("myarr1", {}, {}, { adios2::UnknownDim });
    io.DefineVariable<double>(
      "myarr2", { ncells * mpi_size }, { ncells * mpi_rank }, { ncells });

    auto writer = io.Open("test.h5", adios2::Mode::Write);

    for (auto t { 0 }; t < nsteps; ++t) {
      if (mpi_rank == 0) {
        printf("Timestep #%d\n\n", t);
      }
      const auto nprt = (std::size_t)((t + 1) * (mpi_rank + 1));
      if (nprt > maxptl) {
        throw std::runtime_error("Too many particles");
      }
      adios.EnterComputationBlock();
      Kokkos::parallel_for(
        "fill_myarr1", nprt, KOKKOS_LAMBDA(const int i) {
          myarr1(i) += static_cast<double>(i);
        });
      Kokkos::parallel_for(
        "fill_myarr2", ncells, KOKKOS_LAMBDA(const int i) {
          myarr2(i) += static_cast<double>(i) + 0.1 * t;
        });
      adios.ExitComputationBlock();
      writer.BeginStep();

      // write myarr1
      auto myarr1_slice = Kokkos::View<double*>("myarr1_slice", nprt);
      auto slice        = std::make_pair((std::size_t)0, (std::size_t)nprt);
      Kokkos::deep_copy(myarr1_slice, Kokkos::subview(myarr1, slice));

      auto var1 = io.InquireVariable<double>("myarr1");
      var1.SetSelection({ {}, { nprt } });
      writer.Put<double>(var1, myarr1_slice);

      // write myarr2
      auto var2 = io.InquireVariable<double>("myarr2");
      writer.Put<double>(var2, myarr2);

      writer.EndStep();
    }
    if (mpi_rank == 0) {
      std::cout << "Closing file\n";
    }
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
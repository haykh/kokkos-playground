#include <adios2.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
#ifndef ADIOS2_USE_MPI
  throw std::runtime_error("This example requires MPI");
#endif

  int rank  = 0;
  int nproc = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const int NSTEPS = 5;
  srand(rank * 32767);

  adios2::ADIOS       adios(MPI_COMM_WORLD);

  const size_t        Nglobal = 6;
  std::vector<double> v0(Nglobal);

  unsigned int        Nelems;
  std::vector<double> v2;

  try {
    adios2::IO io = adios.DeclareIO("Output");
    io.SetEngine("HDF5");
    io.SetParameters({
      {"verbose", "4"}
    });

    adios2::Variable<double> varV0 = io.DefineVariable<double>("v0", {}, {}, { Nglobal });
    adios2::Variable<double> varV1
      = io.DefineVariable<double>("v1", { nproc * Nglobal }, { rank * Nglobal }, { Nglobal });
    adios2::Variable<double> varV2
      = io.DefineVariable<double>("v2", {}, {}, { adios2::UnknownDim });

    adios2::Engine writer = io.Open("localArray.h5", adios2::Mode::Write);

    for (int step = 0; step < NSTEPS; step++) {
      writer.BeginStep();
      for (size_t i = 0; i < Nglobal; i++) {
        v0[i] = rank * 1.0 + step * 0.1;
      }
      writer.Put<double>(varV0, v0.data());
      writer.Put<double>(varV1, v0.data());

      Nelems = rand() % 6 + 5;
      v2.reserve(Nelems);
      for (size_t i = 0; i < Nelems; i++) {
        v2[i] = rank * 1.0 + step * 0.1;
      }
      varV2.SetSelection(adios2::Box<adios2::Dims>({}, { Nelems }));
      writer.Put<double>(varV2, v2.data());

      writer.EndStep();
    }

    writer.Close();
  } catch (std::exception& e) {
    std::cout << "ERROR: ADIOS2 exception: " << e.what() << std::endl;
  }
  MPI_Finalize();
  return 0;
}

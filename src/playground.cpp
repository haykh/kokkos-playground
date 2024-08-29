#include "playground.hpp"

#include <Kokkos_Core.hpp>

#ifdef ADIOS2_ENABLED
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
#endif

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#include <iostream>
#include <string>

void Write(adios2::ADIOS&            adios,
           const std::string&        fname,
           const std::string&        engine,
           Kokkos::View<float** [6]> em,
           Kokkos::View<int*>        I,
           Kokkos::View<short*>      tag,
           std::size_t               npart) {
  std::vector<std::size_t> g_shape;
  std::vector<std::size_t> l_corner;
  std::vector<std::size_t> l_shape;
  std::size_t              g_npart;
  std::size_t              l_shiftnpart;
  std::size_t              l_npart = npart;
#if defined(MPI_ENABLED)
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // pad fields in one direction
  g_shape.push_back(em.extent(0) * mpi_size);
  l_corner.push_back(em.extent(0) * mpi_rank);
  l_shape.push_back(em.extent(0));

  g_shape.push_back(em.extent(1));
  l_corner.push_back(0);
  l_shape.push_back(em.extent(1));

  g_shape.push_back(6);
  l_corner.push_back(0);
  l_shape.push_back(6);

  std::vector<std::size_t> nparts(mpi_size);
  MPI_Allgather(&l_npart,
                1,
                MPI_UNSIGNED_LONG_LONG,
                nparts.data(),
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_COMM_WORLD);

  for (auto i { 0 }; i < mpi_size; ++i) {
    if (i < mpi_rank) {
      l_shiftnpart += nparts[i];
    }
    g_npart += nparts[i];
  }

#else
  int mpi_rank = 0, mpi_size = 1;
  for (auto d { 0u }; d < 2; ++d) {
    g_shape.push_back(em.extent(d));
    l_corner.push_back(0);
    l_shape.push_back(em.extent(d));
  }
  g_shape.push_back(6);
  l_corner.push_back(0);
  l_shape.push_back(6);

  g_npart      = l_npart;
  l_shiftnpart = 0;
#endif

  adios2::IO wrtIO = adios.DeclareIO("WriteIO");
  wrtIO.SetEngine(engine);

  auto emVar    = wrtIO.DefineVariable<float>("em", g_shape, l_corner, l_shape);
  auto npartVar = wrtIO.DefineVariable<std::size_t>("npart",
                                                    { adios2::UnknownDim },
                                                    { adios2::UnknownDim },
                                                    { adios2::UnknownDim });
  auto IVar     = wrtIO.DefineVariable<int>("I",
                                        { adios2::UnknownDim },
                                        { adios2::UnknownDim },
                                        { adios2::UnknownDim });
  auto tagVar   = wrtIO.DefineVariable<short>("tag",
                                            { adios2::UnknownDim },
                                            { adios2::UnknownDim },
                                            { adios2::UnknownDim });

  adios2::Engine wrtWriter = wrtIO.Open(fname, adios2::Mode::Write);

  wrtWriter.BeginStep();

  wrtWriter.Put(emVar, em.data());

  npartVar.SetShape({ static_cast<std::size_t>(mpi_size) });
  npartVar.SetSelection(
    adios2::Box<adios2::Dims>({ static_cast<std::size_t>(mpi_rank) }, { 1 }));
  wrtWriter.Put(npartVar, &l_npart);

  IVar.SetShape({ g_npart });
  IVar.SetSelection(adios2::Box<adios2::Dims>({ l_shiftnpart }, { l_npart }));
  wrtWriter.Put(IVar, I.data());

  tagVar.SetShape({ g_npart });
  tagVar.SetSelection(adios2::Box<adios2::Dims>({ l_shiftnpart }, { l_npart }));
  wrtWriter.Put(tagVar, tag.data());

  wrtWriter.EndStep();

  wrtWriter.Close();
}

void Read(adios2::ADIOS&     adios,
          const std::string& fname,
          const std::string& engine,
          std::size_t        nx,
          std::size_t        ny,
          std::size_t        maxnpart) {
  std::vector<std::size_t>  g_shape;
  std::vector<std::size_t>  l_corner;
  std::vector<std::size_t>  l_shape;
  adios2::Box<adios2::Dims> sel_em;
#if defined(MPI_ENABLED)
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  sel_em.first.push_back(nx * mpi_rank);
  sel_em.second.push_back(nx);

  sel_em.first.push_back(0);
  sel_em.second.push_back(ny);

  sel_em.first.push_back(0);
  sel_em.second.push_back(6);
#else
  int mpi_rank = 0, mpi_size = 1;
  sel_em.first.push_back(0);
  sel_em.second.push_back(nx);

  sel_em.first.push_back(0);
  sel_em.second.push_back(ny);

  sel_em.first.push_back(0);
  sel_em.second.push_back(6);
#endif
  Kokkos::View<float** [6]> em("em", nx, ny);
  Kokkos::View<int*>        I("I", maxnpart);
  Kokkos::View<short*>      tag("tag", maxnpart);
  std::size_t               npart;

  adios2::IO rdIO = adios.DeclareIO("ReadIO");
  rdIO.SetEngine(engine);

  adios2::Engine rdReader = rdIO.Open(fname, adios2::Mode::Read);

  auto emVar = rdIO.InquireVariable<float>("em");
  emVar.SetSelection(sel_em);
  rdReader.Get(emVar, em.data());

  auto npartVar = rdIO.InquireVariable<std::size_t>("npart");
  npartVar.SetSelection(
    adios2::Box<adios2::Dims> { { static_cast<std::size_t>(mpi_rank) }, { 1 } });
  rdReader.Get(npartVar, &npart);

  std::size_t l_shiftnpart;
  std::size_t l_npart = npart;
#if defined(MPI_ENABLED)
  std::vector<std::size_t> nparts(mpi_size);
  MPI_Allgather(&l_npart,
                1,
                MPI_UNSIGNED_LONG_LONG,
                nparts.data(),
                1,
                MPI_UNSIGNED_LONG_LONG,
                MPI_COMM_WORLD);
  for (auto i { 0 }; i < mpi_rank; ++i) {
    l_shiftnpart += nparts[i];
  }
#else
  l_shiftnpart = 0;
#endif

  auto IVar = rdIO.InquireVariable<int>("I");
  IVar.SetSelection(adios2::Box<adios2::Dims> { { l_shiftnpart }, { l_npart } });
  rdReader.Get(IVar, I.data());

  auto tagVar = rdIO.InquireVariable<short>("tag");
  tagVar.SetSelection(adios2::Box<adios2::Dims> { { l_shiftnpart }, { l_npart } });
  rdReader.Get(tagVar, tag.data());
}

void Playground(const std::string& action) {
#if defined(MPI_ENABLED)
  adios2::ADIOS adios(MPI_COMM_WORLD);
#else
  adios2::ADIOS adios;
#endif

  const auto engine = "hdf5";
  const auto fname  = "checkpoint.bp";

  if (action == "read") {
    Read(adios, fname, engine, 100, 20, 1000);
  } else {

    Kokkos::View<float** [6]> em("em", 100, 20);
    Kokkos::View<int*>        I("I", 1000);
    Kokkos::View<short*>      tag("tag", 1000);
    const std::size_t         npart = 256;

    Kokkos::parallel_for(
      "Init",
      npart,
      KOKKOS_LAMBDA(std::size_t p) {
        I(p)   = p;
        tag(p) = (p % 2) + 10;
      });

    Write(adios, fname, engine, em, I, tag, npart);
  }
}

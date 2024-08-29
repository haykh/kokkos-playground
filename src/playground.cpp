#include "playground.hpp"

#include <Kokkos_Core.hpp>

#ifdef ADIOS2_ENABLED
  #include <adios2.h>
#endif

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if SIZE_MAX == UCHAR_MAX
  #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
  #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
  #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
  #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
  #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#endif

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
  std::size_t              g_npart      = 0;
  std::size_t              l_shiftnpart = 0;
  std::size_t              l_npart      = npart;
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

  std::size_t* nparts = new std::size_t[mpi_size];
  MPI_Allgather(&l_npart, 1, MPI_SIZE_T, nparts, 1, MPI_SIZE_T, MPI_COMM_WORLD);

  for (auto i { 0 }; i < mpi_size; ++i) {
    if (i < mpi_rank) {
      l_shiftnpart += nparts[i];
    }
    g_npart += nparts[i];
  }
  delete[] nparts;

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

  auto em_h = Kokkos::create_mirror_view(em);
  Kokkos::deep_copy(em_h, em);
  wrtWriter.Put(emVar, em_h.data());

  npartVar.SetShape({ static_cast<std::size_t>(mpi_size) });
  npartVar.SetSelection(
    adios2::Box<adios2::Dims>({ static_cast<std::size_t>(mpi_rank) }, { 1 }));
  wrtWriter.Put(npartVar, &l_npart);

  const auto slice = std::pair<std::size_t, std::size_t> { 0, l_npart };

  IVar.SetShape({ g_npart });
  IVar.SetSelection(adios2::Box<adios2::Dims>({ l_shiftnpart }, { l_npart }));
  auto I_h = Kokkos::create_mirror_view(I);
  Kokkos::deep_copy(I_h, I);
  auto I_sub = Kokkos::subview(I_h, slice);
  wrtWriter.Put(IVar, I_sub.data());

  tagVar.SetShape({ g_npart });
  tagVar.SetSelection(adios2::Box<adios2::Dims>({ l_shiftnpart }, { l_npart }));
  auto tag_h = Kokkos::create_mirror_view(tag);
  Kokkos::deep_copy(tag_h, tag);
  auto tag_sub = Kokkos::subview(tag_h, slice);
  wrtWriter.Put(tagVar, tag_sub.data());

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

  adios2::IO rdIO = adios.DeclareIO("ReadIO");
  rdIO.SetEngine(engine);

  adios2::Engine rdReader = rdIO.Open(fname, adios2::Mode::Read);
  rdReader.BeginStep();

  auto emVar = rdIO.InquireVariable<float>("em");
  emVar.SetSelection(sel_em);
  auto em_h = Kokkos::create_mirror_view(em);
  rdReader.Get(emVar, em_h.data());
  Kokkos::deep_copy(em, em_h);

  // THIS SHOULD BE READ FROM THE FILE (CURRENTLY HARDCODED)
  // std::size_t npart;
  // auto        npartVar = rdIO.InquireVariable<std::size_t>("npart");
  // npartVar.SetSelection(
  //   adios2::Box<adios2::Dims>({ static_cast<std::size_t>(mpi_rank) }, { 1 }));
  // rdReader.Get(npartVar, &npart);
  std::size_t npart = 256;

  std::size_t l_shiftnpart = 0;
  std::size_t l_npart      = npart;
#if defined(MPI_ENABLED)
  std::size_t* nparts = new std::size_t[mpi_size];
  MPI_Allgather(&l_npart, 1, MPI_SIZE_T, nparts, 1, MPI_SIZE_T, MPI_COMM_WORLD);
  for (auto i { 0 }; i < mpi_rank; ++i) {
    l_shiftnpart += nparts[i];
  }
  delete[] nparts;
#else
  l_shiftnpart = 0;
#endif

  const auto slice = std::pair<std::size_t, std::size_t> { 0, l_npart };
  printf("Rank %d: npart = %lu l_shiftnpart = %lu\n", mpi_rank, l_npart, l_shiftnpart);

  auto IVar = rdIO.InquireVariable<int>("I");
  IVar.SetSelection(adios2::Box<adios2::Dims>({ l_shiftnpart }, { l_npart }));
  auto I_h = Kokkos::create_mirror_view(I);
  rdReader.Get(IVar, Kokkos::subview(I_h, slice).data());
  Kokkos::deep_copy(Kokkos::subview(I, slice), Kokkos::subview(I_h, slice));

  auto tagVar = rdIO.InquireVariable<short>("tag");
  tagVar.SetSelection(adios2::Box<adios2::Dims>({ l_shiftnpart }, { l_npart }));
  auto tag_h = Kokkos::create_mirror_view(tag);
  rdReader.Get(tagVar, Kokkos::subview(tag_h, slice).data());
  Kokkos::deep_copy(Kokkos::subview(tag, slice), Kokkos::subview(tag_h, slice));

  printf("Rank %d: em(0 0 0, 5 5 1, 10 1 5) = %f %f %f : I(0, 1, 2) = %d %d "
         "%d : tag(0, 1, 2) = %d %d %d \n",
         mpi_rank,
         em_h(0, 0, 0),
         em_h(5, 5, 1),
         em_h(10, 1, 5),
         I_h(0),
         I_h(1),
         I_h(2),
         tag_h(0),
         tag_h(1),
         tag_h(2));

  rdReader.EndStep();
  rdReader.Close();
}

void Playground(const std::string& action, const std::string& engine) {
#if defined(MPI_ENABLED)
  adios2::ADIOS adios(MPI_COMM_WORLD);
#else
  adios2::ADIOS adios;
#endif

  const auto fname = "checkpoint." + engine;

  if (action == "read") {
    Read(adios, fname, engine, 100, 20, 1000);
  } else {

    Kokkos::View<float** [6]> em("em", 100, 20);
    Kokkos::View<int*>        I("I", 1000);
    Kokkos::View<short*>      tag("tag", 1000);
    const std::size_t         npart = 256;

    Kokkos::parallel_for(
      "InitEM",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { 100, 20 }),
      KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
        em(i, j, 0) = (i * 20 + j) + 2000;
        em(i, j, 1) = (i * 20 + j) + 5000;
        em(i, j, 5) = (i * 20 + j) + 8000;
      });

    Kokkos::parallel_for(
      "InitP",
      npart,
      KOKKOS_LAMBDA(std::size_t p) {
        I(p)   = p;
        tag(p) = (p % 2) + 10;
      });

    Write(adios, fname, engine, em, I, tag, npart);
  }
}

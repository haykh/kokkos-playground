add_executable(main.xc main.cpp)

if(${adios2})
  if(${mpi})
    target_link_libraries(main.xc adios2::cxx11_mpi)
  else()
    target_link_libraries(main.xc adios2::cxx11)
  endif()
endif()

if(${mpi})
  target_link_libraries(main.xc MPI::MPI_CXX)
  target_include_directories(main.xc PUBLIC ${MPI_CXX_INCLUDE_PATH})
endif()

# Kokkos version

add_executable(main-kokkos.xc main_kokkos.cpp)
target_link_libraries(main-kokkos.xc Kokkos::kokkos)

if(${adios2})
  if(${mpi})
    target_link_libraries(main-kokkos.xc adios2::cxx11_mpi)
  else()
    target_link_libraries(main-kokkos.xc adios2::cxx11)
  endif()
endif()

if(${mpi})
  target_link_libraries(main-kokkos.xc MPI::MPI_CXX)
  target_include_directories(main-kokkos.xc PUBLIC ${MPI_CXX_INCLUDE_PATH})
endif()
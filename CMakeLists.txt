cmake_minimum_required(VERSION 3.16)
project(Kokkos-Adios2-MPI-Boilerplate LANGUAGES CXX C)

if(${Kokkos_ENABLE_CUDA})
  enable_language(CUDA)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED)

if(${adios2})
  find_package(adios2 REQUIRED)
endif()

if(${mpi})
  find_package(MPI REQUIRED)
endif()

add_subdirectory(src)